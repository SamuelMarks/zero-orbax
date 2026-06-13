import json
import inspect
import types
import orbax.checkpoint as ocp


def dump_api(module):
    api = {}
    for name in dir(module):
        if name.startswith("_") and name not in ("__version__",):
            continue
        obj = getattr(module, name)

        if inspect.isclass(obj):
            methods = []
            for m_name in dir(obj):
                if not m_name.startswith("_") and callable(getattr(obj, m_name)):
                    methods.append(m_name)
            api[name] = {"type": "class", "methods": methods}
        elif isinstance(obj, types.ModuleType):
            api[name] = {"type": "namespace"}
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "other"}
    return api


snapshot = dump_api(ocp)
with open("tests/orbax_api_snapshot.json", "w") as f:
    json.dump(snapshot, f, indent=2)

print("Snapshot generated.")
