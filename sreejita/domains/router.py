from sreejita.domains import retail, ecommerce, customer, text

DOMAIN_REGISTRY = {
    "retail": retail,
    "ecommerce": ecommerce,
    "customer": customer,
    "text": text,
    "generic": None
}

def apply_domain(df, domain_name: str):
    domain = DOMAIN_REGISTRY.get(domain_name, None)

    if domain is None:
        return df

    if hasattr(domain, "enrich"):
        return domain.enrich(df)

    return df
