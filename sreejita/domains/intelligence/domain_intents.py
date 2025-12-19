# sreejita/domains/intelligence/domain_intents.py

DOMAIN_INTENTS = {

    # =====================================================
    # HEALTHCARE
    # =====================================================
    "healthcare": {
        "high": {
            "diagnosis", "patient", "treatment", "admission",
            "discharge", "doctor", "hospital", "clinical",
            "readmitted", "readmission", "patient_id",
            "outcome_score", "length_of_stay", "mortality",
            "insurance_provider", "blood_type"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "status", "notes"
        }
    },

    # =====================================================
    # MARKETING
    # =====================================================
    "marketing": {
        "high": {
            "campaign", "impressions", "clicks", "ctr",
            "ad_spend", "roas", "cpc", "audience",
            "creative", "converted", "campaign_id"
        },
        "ambiguous": {
            "channel", "source", "conversion",
            "cost", "revenue", "region"
        }
    },

    # =====================================================
    # OPERATIONS / SUPPLY CHAIN
    # =====================================================
    "ops": {
        "high": {
            "warehouse", "carrier", "shipping_mode",
            "delivery_status", "inventory_level",
            "restock", "supplier", "processing_time",
            "lead_time", "cycle_time", "on_time_delivery"
        },
        "ambiguous": {
            "ship_date", "order_id", "quantity",
            "location", "return"
        }
    },

    # =====================================================
    # FINANCE
    # =====================================================
    "finance": {
        "high": {
            "portfolio", "asset", "liability", "equity",
            "dividend", "interest_rate", "ticker",
            "balance", "net_income", "cash_flow",
            "ebitda", "volatility", "beta"
        },
        "ambiguous": {
            "amount", "date", "account",
            "currency", "value", "revenue", "profit"
        }
    },

    # =====================================================
    # RETAIL
    # =====================================================
    "retail": {
        "high": {
            "store_id", "store_location", "cashier",
            "pos_id", "shelf", "aisle", "zone"
        },
        "ambiguous": {
            "store", "sales", "category",
            "item", "quantity", "profit"
        }
    },

    # =====================================================
    # CUSTOMER (CRM / CX)
    # =====================================================
    "customer": {
        "high": {
            "customer_id", "customer_name", "email",
            "phone", "segment", "recency",
            "frequency", "monetary",
            "lifetime_value", "churn", "engagement"
        },
        "ambiguous": {
            "purchase_date", "location",
            "device", "channel",
            "transaction_id", "amount"
        }
    },

    # =====================================================
    # ECOMMERCE
    # =====================================================
    "ecommerce": {
        "high": {
            "conversion_rate", "cart_abandonment",
            "aov", "cac", "ltv", "roi",
            "cart_size", "checkout_completion",
            "return_rate"
        },
        "ambiguous": {
            "session_id", "referrer", "ip_address",
            "shipping_address", "payment_gateway",
            "discount_code"
        }
    },

    # =====================================================
    # HR / WORKFORCE  🔥 NEW (MANDATORY)
    # =====================================================
    "hr": {
        "high": {
            # Identity
            "employee", "employee_id", "staff",

            # Org structure
            "department", "designation", "role", "manager",

            # Compensation
            "salary", "compensation", "ctc",
            "payroll", "bonus",

            # Lifecycle
            "attrition", "termination",
            "resignation", "joining",
            "hire", "exit",

            # Performance & attendance
            "performance", "rating",
            "leave", "absence", "attendance"
        },
        "ambiguous": {
            "id", "date", "status",
            "gender", "age", "location"
        }
    },
} 
