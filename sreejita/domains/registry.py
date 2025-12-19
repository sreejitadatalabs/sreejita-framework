"""
Domain Registry (v2.x FINAL)

This file defines the SINGLE registry instance used
across the entire framework.
"""

class DomainRegistry:
    def __init__(self):
        self._domains = {}

    def register(self, name, domain_cls, detector_cls=None):
        self._domains[name] = domain_cls

    def get_domain(self, name):
        domain_cls = self._domains.get(name)
        if not domain_cls:
            return None
        return domain_cls()

    def list_domains(self):
        return list(self._domains.keys())


# 🔒 SINGLETON REGISTRY (THIS MUST BE IMPORTED EVERYWHERE)
registry = DomainRegistry()
