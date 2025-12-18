DOMAIN_INTENTS = {
    "healthcare": {
        "high": {
            "diagnosis", "patient", "treatment", "admission",
            "discharge", "doctor", "hospital", "clinical",
            "insurance_provider", "blood_type"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "status", "notes"
        }
    },

    "marketing": {
        "high": {
            "campaign", "impressions", "clicks", "ctr",
            "ad_spend", "roas", "cpc", "audience", "creative"
        },
        "ambiguous": {
            "channel", "source", "conversion", "cost",
            "revenue", "region"
        }
    },

    "ops": {
        "high": {
            "warehouse", "carrier", "shipping_mode",
            "delivery_status", "inventory_level", "restock",
            "supplier", "processing_time", "lead_time", "weight"
        },
        "ambiguous": {
            "ship_date", "order_id", "quantity",
            "return", "location"
        }
    },

    "finance": {
        "high": {
            "portfolio", "asset", "liability", "equity",
            "dividend", "interest_rate", "ticker",
            "transaction_type", "balance"
        },
        "ambiguous": {
            "amount", "date", "account",
            "currency", "value", "revenue", "profit"
        }
    },

    "retail": {
        "high": {
            "store_id", "store_location", "cashier",
            "pos_id", "shelf", "aisle", "floor", "zone"
        },
        "ambiguous": {
            "store", "sales", "manager",
            "category", "item", "quantity"
        }
    },
}
