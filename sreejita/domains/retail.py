from sreejita.domains.base import BaseDomainDetector


class RetailDomainDetector(BaseDomainDetector):
    domain = "retail"

    # Retail-specific structure (strong)
    PRIMARY_SIGNALS = {
        "store_id",
        "pos_id",
        "aisle",
        "shelf",
        "cashier",
    }

    # Product-level context (medium)
    SECONDARY_SIGNALS = {
        "product",
        "item",
        "brand",
        "category",
        "sub_category",
    }

    # Generic business metrics (weak)
    GENERIC_SIGNALS = {
        "sales",
        "revenue",
        "profit",
        "quantity",
        "discount",
        "price",
        "order_id",
        "date",
    }