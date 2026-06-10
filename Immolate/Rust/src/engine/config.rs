use crate::engine::tables::{
    POOL_COMMON, POOL_RARE, POOL_UNCOMMON, is_ante1_locked_tag, is_soulable_pack, pack_info,
    target_joker_pools,
};
use crate::filters::{FilterConfig, JokerLocation};
use crate::item::Item;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelShape {
    NoMatch,
    NoFilter,
    TagOnly,
    VoucherOnly,
    PackOnly,
    Observatory,
    ShopJoker,
    PackJoker,
    AnyJoker,
    Souls,
    Perkeo,
    Erratic,
    Composite,
    Generic,
}

#[derive(Clone, Copy, Debug)]
pub struct CompiledFilter {
    pub raw: FilterConfig,
    pub shape: KernelShape,
    pub wants_joker_shop: bool,
    pub wants_joker_pack: bool,
    pub target_joker_pools: u8,
    pub needs_packs: bool,
    pub selected_soulable_pack: bool,
}

impl CompiledFilter {
    pub fn compile(raw: &FilterConfig) -> Self {
        let wants_joker = raw.joker != Item::RETRY;
        let wants_joker_shop =
            wants_joker && matches!(raw.joker_location, JokerLocation::Shop | JokerLocation::Any);
        let wants_joker_pack =
            wants_joker && matches!(raw.joker_location, JokerLocation::Pack | JokerLocation::Any);
        let needs_packs = raw.pack != Item::RETRY
            || raw.observatory
            || raw.perkeo
            || raw.souls > 0
            || wants_joker_pack;

        Self {
            raw: *raw,
            shape: classify(
                raw,
                wants_joker_shop,
                wants_joker_pack,
                target_joker_pools(raw.joker),
            ),
            wants_joker_shop,
            wants_joker_pack,
            target_joker_pools: target_joker_pools(raw.joker),
            needs_packs,
            selected_soulable_pack: raw.pack == Item::RETRY || is_soulable_pack(raw.pack),
        }
    }

    pub const fn chunk_size(self) -> i64 {
        match self.shape {
            KernelShape::Erratic => 16_384,
            KernelShape::ShopJoker | KernelShape::PackJoker | KernelShape::AnyJoker => 32_768,
            KernelShape::Composite => 4_096,
            KernelShape::Generic => 262_144,
            _ => 65_536,
        }
    }
}

fn classify(
    raw: &FilterConfig,
    wants_joker_shop: bool,
    wants_joker_pack: bool,
    target_joker_pools: u8,
) -> KernelShape {
    let has_tags = raw.tag1 != Item::RETRY || raw.tag2 != Item::RETRY;
    let has_voucher = raw.voucher != Item::RETRY;
    let has_pack = raw.pack != Item::RETRY;
    let has_joker = raw.joker != Item::RETRY;
    let has_souls = raw.souls > 0;
    let has_erratic = raw.erratic;

    if is_static_no_match(
        raw,
        has_joker,
        wants_joker_shop,
        wants_joker_pack,
        target_joker_pools,
    ) {
        return KernelShape::NoMatch;
    }

    if !has_tags
        && !has_voucher
        && !has_pack
        && !raw.observatory
        && !has_joker
        && !has_souls
        && !raw.perkeo
        && !has_erratic
    {
        return KernelShape::NoFilter;
    }
    if has_tags {
        return if has_voucher
            || has_pack
            || raw.observatory
            || has_joker
            || has_souls
            || raw.perkeo
            || has_erratic
        {
            KernelShape::Composite
        } else {
            KernelShape::TagOnly
        };
    }
    if has_voucher {
        return if has_pack || raw.observatory || has_joker || has_souls || raw.perkeo || has_erratic
        {
            KernelShape::Composite
        } else {
            KernelShape::VoucherOnly
        };
    }
    if has_erratic {
        return if has_pack || raw.observatory || has_joker || has_souls || raw.perkeo {
            KernelShape::Composite
        } else {
            KernelShape::Erratic
        };
    }
    if raw.observatory {
        return if has_pack || has_joker || has_souls || raw.perkeo {
            KernelShape::Composite
        } else {
            KernelShape::Observatory
        };
    }
    if has_joker {
        if has_souls || raw.perkeo {
            return KernelShape::Composite;
        }
        if has_pack && wants_joker_shop && !wants_joker_pack {
            return KernelShape::Composite;
        }
        return if wants_joker_shop && wants_joker_pack {
            KernelShape::AnyJoker
        } else if wants_joker_shop {
            KernelShape::ShopJoker
        } else {
            KernelShape::PackJoker
        };
    }
    if has_souls {
        return if raw.perkeo {
            KernelShape::Composite
        } else {
            KernelShape::Souls
        };
    }
    if raw.perkeo {
        return KernelShape::Perkeo;
    }
    if has_pack {
        return KernelShape::PackOnly;
    }
    KernelShape::Generic
}

fn is_static_no_match(
    raw: &FilterConfig,
    has_joker: bool,
    wants_joker_shop: bool,
    wants_joker_pack: bool,
    target_joker_pools: u8,
) -> bool {
    if has_joker
        && (wants_joker_shop || wants_joker_pack)
        && target_joker_pools & (POOL_COMMON | POOL_UNCOMMON | POOL_RARE) == 0
    {
        return true;
    }

    if is_ante1_locked_tag(raw.tag1) || is_ante1_locked_tag(raw.tag2) {
        return true;
    }

    if raw.souls > 0 {
        if raw.pack != Item::RETRY {
            return !is_soulable_pack(raw.pack) || raw.souls > pack_info(raw.pack).size as i64;
        }
        if raw.souls > 5 {
            return true;
        }
    }

    if raw.perkeo && raw.pack != Item::RETRY && !is_soulable_pack(raw.pack) {
        return true;
    }

    raw.erratic && raw.no_faces && raw.min_face_cards > 0
}
