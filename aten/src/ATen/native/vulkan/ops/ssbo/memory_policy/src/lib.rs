#![forbid(unsafe_op_in_unsafe_fn)]

use core::cmp::min;

const NO_CAP: u64 = u64::MAX;
const WEIGHT_KIND: u32 = 1;
const POOL_KIND: u32 = 2;
const RESIDENT_KIND: u32 = 3;
const TOPOLOGY_UMA: u32 = 1;
const TOPOLOGY_DISCRETE: u32 = 2;
const TOPOLOGY_MIXED: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VkssboMemorySnapshot {
    pub heap_budget: u64,
    pub heap_usage: u64,
    pub weight_cache_bytes: u64,
    pub transient_pool_bytes: u64,
    pub resident_bytes: u64,
    pub safety_reserve: u64,
    pub fallback_weight_limit: u64,
    pub fallback_pool_limit: u64,
    pub weight_cache_cap: u64,
    pub transient_pool_cap: u64,
    pub requested_bytes: u64,
    pub request_kind: u32,
    pub memory_topology: u32,
    pub has_device_local_only: u32,
    pub has_host_visible_device_local: u32,
    pub has_host_visible_system: u32,
    pub device_local_budget: u64,
    pub device_local_usage: u64,
    pub device_local_only_budget: u64,
    pub device_local_only_usage: u64,
    pub host_visible_device_local_budget: u64,
    pub host_visible_device_local_usage: u64,
    pub host_visible_system_budget: u64,
    pub host_visible_system_usage: u64,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VkssboMemoryPlan {
    pub safe_budget: u64,
    pub available_bytes: u64,
    pub weight_cache_limit: u64,
    pub transient_pool_limit: u64,
    pub resident_limit: u64,
    pub non_managed_bytes: u64,
    pub dynamic_budget: u32,
    pub allow_new_allocation: u32,
    pub pressure_permille: u32,
    pub recommended_evict_kind: u32,
    pub memory_topology: u32,
    pub prefer_host_visible: u32,
    pub prefer_device_local: u32,
    pub reserved: u32,
}

fn capped(value: u64, cap: u64) -> u64 {
    if cap == NO_CAP { value } else { min(value, cap) }
}

fn safe_budget(snapshot: &VkssboMemorySnapshot) -> (u64, bool) {
    if snapshot.heap_budget == 0 {
        return (NO_CAP, false);
    }
    // Keep the same 80% policy as the existing C++ implementation, then
    // reserve a fixed amount for Vulkan/driver bookkeeping.
    let eighty_percent = snapshot.heap_budget.saturating_mul(4) / 5;
    (eighty_percent.saturating_sub(snapshot.safety_reserve), true)
}

fn remaining_for(limit: u64, usage: u64, tracked: u64) -> u64 {
    limit.saturating_sub(usage.saturating_sub(tracked))
}

fn requested_room(limit: u64, tracked: u64, requested: u64) -> bool {
    requested <= limit.saturating_sub(tracked)
}

fn pressure_permille(usage: u64, safe: u64, dynamic: bool) -> u32 {
    if !dynamic || safe == 0 { return 0; }
    let ratio = usage.saturating_mul(1000) / safe;
    min(ratio, 2000) as u32
}

fn compute(snapshot: &VkssboMemorySnapshot) -> VkssboMemoryPlan {
    let (safe, dynamic) = safe_budget(snapshot);
    let usage = snapshot.heap_usage;

    let managed_total = snapshot
        .weight_cache_bytes
        .saturating_add(snapshot.transient_pool_bytes)
        .saturating_add(snapshot.resident_bytes);
    let non_managed = usage.saturating_sub(managed_total);

    let weight = if dynamic {
        capped(remaining_for(safe, usage, snapshot.weight_cache_bytes),
               snapshot.weight_cache_cap)
    } else {
        capped(snapshot.fallback_weight_limit, snapshot.weight_cache_cap)
    };
    let pool = if dynamic {
        capped(remaining_for(safe, usage, snapshot.transient_pool_bytes),
               snapshot.transient_pool_cap)
    } else {
        capped(snapshot.fallback_pool_limit, snapshot.transient_pool_cap)
    };
    let resident = if dynamic {
        remaining_for(safe, usage, snapshot.resident_bytes)
    } else {
        NO_CAP
    };
    let available = if dynamic { safe.saturating_sub(usage) } else { NO_CAP };

    let request_limit = match snapshot.request_kind {
        WEIGHT_KIND => requested_room(weight, snapshot.weight_cache_bytes,
                                      snapshot.requested_bytes),
        POOL_KIND => requested_room(pool, snapshot.transient_pool_bytes,
                                    snapshot.requested_bytes),
        RESIDENT_KIND => requested_room(resident, snapshot.resident_bytes,
                                        snapshot.requested_bytes),
        _ => snapshot.requested_bytes <= available,
    };

    // Under pressure, release reusable transient buffers before evicting
    // persistent weights. If no pool is available, weights are the next
    // eviction candidate; resident tensors are explicit and last to evict.
    let recommended = if dynamic && usage > safe {
        if snapshot.transient_pool_bytes != 0 { POOL_KIND }
        else if snapshot.weight_cache_bytes != 0 { WEIGHT_KIND }
        else if snapshot.resident_bytes != 0 { RESIDENT_KIND }
        else { 0 }
    } else { 0 };

    // Stage A only reports a topology-aware preference. The C++ allocator
    // remains the sole Vulkan resource owner and applies these hints in the
    // following implementation stage after hardware diagnostics are verified.
    let prefer_host_visible = match snapshot.memory_topology {
        TOPOLOGY_UMA => snapshot.has_host_visible_device_local != 0,
        _ => false,
    };
    let prefer_device_local = match snapshot.memory_topology {
        TOPOLOGY_DISCRETE | TOPOLOGY_MIXED => snapshot.has_device_local_only != 0,
        _ => false,
    };

    VkssboMemoryPlan {
        safe_budget: safe,
        available_bytes: available,
        weight_cache_limit: weight,
        transient_pool_limit: pool,
        resident_limit: resident,
        non_managed_bytes: non_managed,
        dynamic_budget: dynamic as u32,
        allow_new_allocation: request_limit as u32,
        pressure_permille: pressure_permille(usage, safe, dynamic),
        recommended_evict_kind: recommended,
        memory_topology: snapshot.memory_topology,
        prefer_host_visible: prefer_host_visible as u32,
        prefer_device_local: prefer_device_local as u32,
        reserved: 0,
    }
}

#[no_mangle]
pub extern "C" fn vkssbo_memory_plan(
    snapshot: *const VkssboMemorySnapshot,
    plan: *mut VkssboMemoryPlan,
) -> i32 {
    if snapshot.is_null() || plan.is_null() { return 0; }
    // The pointers are supplied by the C++ ABI wrapper and are checked above.
    let snapshot = unsafe { &*snapshot };
    let computed = compute(snapshot);
    unsafe { *plan = computed; }
    1
}

#[no_mangle]
pub extern "C" fn vkssbo_memory_policy_version() -> u32 { 1 }

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> VkssboMemorySnapshot {
        VkssboMemorySnapshot {
            heap_budget: 10_000,
            heap_usage: 3_000,
            weight_cache_bytes: 1_000,
            transient_pool_bytes: 500,
            resident_bytes: 250,
            safety_reserve: 500,
            fallback_weight_limit: 8_000,
            fallback_pool_limit: 2_000,
            weight_cache_cap: NO_CAP,
            transient_pool_cap: NO_CAP,
            ..Default::default()
        }
    }

    #[test]
    fn dynamic_limits_exclude_other_usage() {
        let p = compute(&base());
        assert_eq!(p.safe_budget, 7_500);
        assert_eq!(p.weight_cache_limit, 5_500);
        assert_eq!(p.transient_pool_limit, 5_000);
        assert_eq!(p.resident_limit, 4_750);
        assert_eq!(p.non_managed_bytes, 1_250);
    }

    #[test]
    fn caps_are_hard_upper_bounds() {
        let mut s = base();
        s.weight_cache_cap = 2_000;
        s.transient_pool_cap = 1_500;
        let p = compute(&s);
        assert_eq!(p.weight_cache_limit, 2_000);
        assert_eq!(p.transient_pool_limit, 1_500);
    }

    #[test]
    fn low_budget_denies_request() {
        let mut s = base();
        s.heap_usage = 8_000;
        s.request_kind = WEIGHT_KIND;
        s.requested_bytes = 2_001;
        let p = compute(&s);
        assert_eq!(p.allow_new_allocation, 0);
        assert_eq!(p.recommended_evict_kind, POOL_KIND);
    }

    #[test]
    fn fallback_is_used_without_budget() {
        let mut s = base();
        s.heap_budget = 0;
        s.weight_cache_cap = 3_000;
        let p = compute(&s);
        assert_eq!(p.dynamic_budget, 0);
        assert_eq!(p.weight_cache_limit, 3_000);
        assert_eq!(p.transient_pool_limit, 2_000);
        assert_eq!(p.available_bytes, NO_CAP);
    }

    #[test]
    fn generic_request_uses_remaining_safe_budget() {
        let mut s = base();
        s.requested_bytes = 4_000;
        assert_eq!(compute(&s).allow_new_allocation, 1);
        s.requested_bytes = 7_001;
        assert_eq!(compute(&s).allow_new_allocation, 0);
    }

    #[test]
    fn incoming_request_reduces_remaining_limit() {
        let mut s = base();
        s.request_kind = POOL_KIND;
        s.transient_pool_bytes = 2_000;
        // A buffer being released has already been allocated by Vulkan, so
        // its bytes are present in heap_usage while the pool computes its
        // post-release limit.
        s.heap_usage = 4_000;
        s.requested_bytes = 1_000;
        let p = compute(&s);
        assert_eq!(p.transient_pool_limit, 5_500);
        assert_eq!(p.allow_new_allocation, 1);
        s.requested_bytes = 3_501;
        assert_eq!(compute(&s).allow_new_allocation, 0);
    }

    #[test]
    fn pressure_prefers_reusable_pool_eviction() {
        let mut s = base();
        s.heap_usage = 8_000;
        s.transient_pool_bytes = 1_000;
        s.weight_cache_bytes = 500;
        let p = compute(&s);
        assert_eq!(p.recommended_evict_kind, POOL_KIND);
        assert!(p.pressure_permille > 1000);
    }

    #[test]
    fn resident_budget_is_independent_of_weight_cap() {
        let mut s = base();
        s.weight_cache_cap = 100;
        s.request_kind = RESIDENT_KIND;
        s.requested_bytes = 4_000;
        let p = compute(&s);
        assert_eq!(p.allow_new_allocation, 1);
        assert!(p.resident_limit > p.weight_cache_limit);
    }

    #[test]
    fn uma_prefers_host_visible_device_local_memory() {
        let mut s = base();
        s.memory_topology = TOPOLOGY_UMA;
        s.has_host_visible_device_local = 1;
        let p = compute(&s);
        assert_eq!(p.memory_topology, TOPOLOGY_UMA);
        assert_eq!(p.prefer_host_visible, 1);
        assert_eq!(p.prefer_device_local, 0);
    }

    #[test]
    fn discrete_prefers_device_local_only_memory() {
        let mut s = base();
        s.memory_topology = TOPOLOGY_DISCRETE;
        s.has_device_local_only = 1;
        s.has_host_visible_system = 1;
        let p = compute(&s);
        assert_eq!(p.memory_topology, TOPOLOGY_DISCRETE);
        assert_eq!(p.prefer_host_visible, 0);
        assert_eq!(p.prefer_device_local, 1);
    }

    #[test]
    fn mixed_topology_keeps_device_local_preference() {
        let mut s = base();
        s.memory_topology = TOPOLOGY_MIXED;
        s.has_device_local_only = 1;
        s.has_host_visible_device_local = 1;
        let p = compute(&s);
        assert_eq!(p.prefer_host_visible, 0);
        assert_eq!(p.prefer_device_local, 1);
    }
}
