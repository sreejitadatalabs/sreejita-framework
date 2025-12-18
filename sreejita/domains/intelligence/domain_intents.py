DOMAIN_INTENTS = {
    "healthcare": {
        "high": {
            "diagnosis", "patient", "treatment", "admission",
            "discharge", "doctor", "hospital", "clinical", 
            "readmitted", " readmission", "patient_id", 
            "outcome_score", "length_of_stay", "mortality",
            "insurance_provider", "blood_type"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "status", "notes"
        }
    },

    "marketing": {
        "high": {
            "campaign", "impressions", "clicks", "ctr",
            "ad_spend", "roas", "cpc", "audience", "creative", 
            "converted", "campaign_id"
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
            "supplier", "processing_time", "lead_time", "weight",
            "cycle_time", "on_time_delivery"
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
            "transaction_type", "balance", "net_income", 
            "cash_flow", "ebitda", "volatility", "beta"
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
            "category", "item", "quantity", "profit"
        }
    },

    "customer": {
        "high": {
            "customer_id", "customer_name", "email", 
            "phone", "segment", "recency", "frequency", 
            "monetary", "lifetime_value", "churn", "engagement"
        },
        "ambiguous": {
            "purchase_date", "location", "device", 
            "channel", "transaction_id", "amount"
        }
    },

    "ecommerce": {
        "high": {
            "conversion_rate", "cart_abandonment", "aov", "cac", 
            "ltv", "roas", "roi", "cart_size", "checkout_completion", 
            "return_rate"
        },
        "ambiguous": {
            "session_id", "user_agent", "referrer", "ip_address", 
            "shipping_address", "tracking_number", "payment_gateway",
            "discount_code", "cart_session_duration"
        }
    },
}
