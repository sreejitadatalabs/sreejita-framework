from sreejita.domains.base import BaseDomainDetector


class RetailDomainDetector(BaseDomainDetector):
    domain = "retail"

    # Retail should be specific, not generic
    PRIMARY_SIGNALS = {
        "pos_id",
        "store_id",
        "aisle",
        "shelf",
        "cashier",
    }

    SECONDARY_SIGNALS = {
        "product",
        "item",
        "brand",
        "category",
        "sub_category",
    }

    # Generic business metrics — LOW weight
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
