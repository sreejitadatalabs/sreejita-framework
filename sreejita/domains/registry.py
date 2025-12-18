"""
Domain Registry (v2.x FINAL)

There must be exactly ONE registry instance in the system.
All domains register here.
All routers resolve engines from here.
"""

class DomainRegistry:
    def __init__(self):
        self._domains = {}

    def register(self, name, domain_cls):
        self._domains[name.lower()] = domain_cls

    def get_domain(self, name):
        domain_cls = self._domains.get(name.lower())
        if not domain_cls:
            return None
        return domain_cls()

    def list_domains(self):
        return list(self._domains.keys())


# 🔒 SINGLETON REGISTRY
registry = DomainRegistry()
