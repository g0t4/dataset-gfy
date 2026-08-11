import importlib.util
import os

os.environ["HOME"] = os.path.join(os.getcwd(), "fakehome")
trace_dir = os.path.join(os.environ["HOME"], ".local/state/nvim/ask-openai/agents")
os.makedirs(trace_dir, exist_ok=True)
with open(os.path.join(trace_dir, "1700000000-trace.json"), "w") as f:
    f.write('{"marker_field_xyz123": "hello_from_fixture_trace"}')

spec = importlib.util.spec_from_file_location("browser_main", "tools/chat_viewer/browser/__main__.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import rich.table
captured_rows = []
orig_add_row = rich.table.Table.add_row
def spy_add_row(self, *args, **kwargs):
    captured_rows.append(args)
    return orig_add_row(self, *args, **kwargs)
rich.table.Table.add_row = spy_add_row

browser = mod.TraceBrowser("agents")
browser.print_help()

has_j_row = any(len(row) > 0 and row[0] == "j" for row in captured_rows)
print(f"HELP_HAS_J_ROW: {has_j_row}")

browser.on_char(b'j')
