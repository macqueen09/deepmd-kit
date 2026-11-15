
import importlib.machinery, json, os, runpy, sys
source = os.path.realpath(os.environ["DPA4C_SELECTED_SOURCE"])
sys.path.insert(0, source)
finder = None
for item in sys.meta_path:
    if type(item).__name__ != "ScikitBuildRedirectingFinder":
        continue
    keys = getattr(item, "known_source_files", None)
    if isinstance(keys, dict) and any(k == "deepmd" or k.startswith("deepmd.") for k in keys):
        if finder is not None:
            raise SystemExit("ambiguous deepmd ScikitBuildRedirectingFinder in sys.meta_path")
        finder = item
if finder is None:
    raise SystemExit("no deepmd ScikitBuildRedirectingFinder; refusing silent fallback")
index = [position for position, item in enumerate(sys.meta_path)
         if item is importlib.machinery.PathFinder]
if len(index) != 1:
    raise SystemExit("importlib.machinery.PathFinder missing from sys.meta_path")
sys.meta_path.remove(finder)
sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder) + 1, finder)
import deepmd
import deepmd.lib
from deepmd.pt_expt.utils import canonical_graph, vesin_graph_builder


def require(path, boundary, label):
    actual, expected = os.path.realpath(path), os.path.realpath(boundary)
    if actual != expected and not actual.startswith(expected + os.sep):
        raise SystemExit("path probe failed for %s: %s not under %s" % (label, actual, expected))


probe = {"selected_source": source, "deepmd_file": deepmd.__file__,
         "deepmd_lib_file": getattr(deepmd.lib, "__file__", None),
         "canonical_graph_file": canonical_graph.__file__,
         "vesin_graph_builder_file": vesin_graph_builder.__file__,
         "native_prefix": sys.prefix,
         "finder": type(finder).__module__ + "." + type(finder).__name__,
         "meta_path": [type(item).__module__ + "." + type(item).__name__ for item in sys.meta_path]}
require(probe["deepmd_file"], os.path.join(source, "deepmd"), "deepmd")
require(probe["canonical_graph_file"],
        os.path.join(source, "deepmd", "pt_expt", "utils", "canonical_graph.py"), "canonical_graph")
require(probe["vesin_graph_builder_file"],
        os.path.join(source, "deepmd", "pt_expt", "utils", "vesin_graph_builder.py"), "vesin_graph_builder")
with open(os.environ["DPA4C_PROBE_PATH"], "x", encoding="utf-8") as stream:
    stream.write(json.dumps(probe, indent=2) + "\n")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
