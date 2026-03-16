"""
数据库可视化浏览器 — 现代浏览器风格 + 树形侧栏
图形化展示当前 CoSearch 项目配置所指向数据库的所有表内容。

直接运行：
    python db_browser.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pymysql
import json
from datetime import datetime

from config import settings

# ─────────────────────── 颜色系统 ───────────────────────
C = {
    "bg_app":        "#F0F2F5",
    "bg_sidebar":    "#F8F9FB",
    "bg_topbar":     "#1C2B3A",
    "bg_content":    "#FFFFFF",
    "bg_toolbar":    "#F8F9FB",
    "bg_row_even":   "#FFFFFF",
    "bg_row_odd":    "#F7F9FC",
    "bg_row_sel":    "#D6E8FF",
    "bg_badge":      "#EAEDF1",
    "bg_btn":        "#2563EB",
    "bg_btn_hover":  "#1D4ED8",
    "bg_input":      "#FFFFFF",
    "bg_detail":     "#F8F9FB",
    "bg_node_hover": "#EBF3FF",
    "bg_node_sel":   "#DBEAFE",
    "bg_group_hdr":  "#EEF0F3",

    "fg_primary":    "#0F1923",
    "fg_secondary":  "#4A5A6A",
    "fg_tertiary":   "#8A9BAB",
    "fg_topbar":     "#E8EDF2",
    "fg_topbar_sub": "#7A9AB4",
    "fg_badge":      "#5A6A7A",
    "fg_btn":        "#FFFFFF",
    "fg_link":       "#2563EB",
    "fg_header":     "#334A5E",
    "fg_null":       "#A0B0C0",
    "fg_ts":         "#047857",
    "fg_json":       "#7C3AED",
    "fg_num":        "#1A5DA8",
    "fg_node_sel":   "#1D4ED8",
    "fg_group":      "#6A7A8A",

    "border_light":  "#E4E8EE",
    "border_mid":    "#CED6DF",
    "border_input":  "#C8D2DC",
    "border_focus":  "#2563EB",
    "accent":        "#2563EB",
    "accent_light":  "#EBF3FF",
}

# ─────────────────────── 字体系统（楷体，不加粗） ───────────────────────
KAI  = "楷体"
MONO = "Courier New"

F = {
    "topbar":     (KAI, 13),
    "topbar_sub": (KAI, 9),
    "title":      (KAI, 14),
    "normal":     (KAI, 11),
    "small":      (KAI, 10),
    "tiny":       (KAI, 9),
    "mono":       (MONO, 10),
    "section":    (KAI, 9),
    "badge":      (KAI, 9),
    "tag":        (KAI, 11),
    "detail_key": (KAI, 11),
    "detail_val": (MONO, 10),
    "header":     (KAI, 10),
}

# ─────────────────────── 数据库工具 ───────────────────────
def connect():
    return pymysql.connect(
        host=settings.db_host,
        user=settings.db_user,
        passwd=settings.db_password,
        port=settings.db_port,
        db=settings.db_name,
        charset="utf8mb4"
    )

def get_all_tables():
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def get_table_columns(table_name):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DESCRIBE `{table_name}`")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def get_table_rows(table_name, limit=300):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT * FROM `{table_name}` ORDER BY 1 DESC LIMIT {limit}")
            rows = cur.fetchall()
        return rows, total
    finally:
        conn.close()

def get_table_row_count(table_name):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            return cur.fetchone()[0]
    finally:
        conn.close()

TS_COLS   = {"start_ts", "end_ts", "ts", "last_updated_ts", "created_at", "updated_at"}
LONG_COLS = {"utterance", "answer", "summary_text", "search_results",
             "content", "profile_text", "description", "text"}

def fmt_cell(col, value):
    if value is None:
        return "NULL", "null"
    raw = str(value)
    if col in TS_COLS:
        try:
            ts = float(raw)
            if ts > 1_000_000_000:
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M:%S"), "ts"
        except Exception:
            pass
    if raw.strip().startswith(("[", "{")):
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                preview = f"[{len(obj)} 项]  " + ", ".join(str(x) for x in obj[:3])
                return (preview[:80] + "…") if len(preview) > 80 else preview, "json"
            if isinstance(obj, dict):
                preview = json.dumps(obj, ensure_ascii=False)
                return (preview[:80] + "…") if len(preview) > 80 else preview, "json"
        except Exception:
            pass
    try:
        float(raw)
        return raw, "num"
    except ValueError:
        pass
    max_len = 100 if col in LONG_COLS else 60
    if len(raw) > max_len:
        return raw[:max_len - 1] + "…", "normal"
    return raw, "normal"


# ─────────────────────── 行详情弹窗 ───────────────────────
class RowDetailDialog(tk.Toplevel):
    def __init__(self, parent, columns, row):
        super().__init__(parent)
        self.title("行详情")
        self.configure(bg=C["bg_detail"])
        self.resizable(True, True)
        self.minsize(560, 420)
        self.geometry("700x580")
        self._build(columns, row)
        self.grab_set()
        self.focus_force()

    def _build(self, columns, row):
        titlebar = tk.Frame(self, bg=C["bg_topbar"], height=44)
        titlebar.pack(fill="x")
        titlebar.pack_propagate(False)
        tk.Label(titlebar, text="行详情", font=F["topbar"],
                 bg=C["bg_topbar"], fg=C["fg_topbar"]).pack(side="left", padx=16)
        tk.Label(titlebar, text=f"{len(columns)} 个字段",
                 font=F["topbar_sub"], bg=C["bg_topbar"],
                 fg=C["fg_topbar_sub"]).pack(side="right", padx=16)

        outer = tk.Frame(self, bg=C["bg_detail"])
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=C["bg_detail"],
                           highlightthickness=0, bd=0, yscrollincrement=4)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner = tk.Frame(canvas, bg=C["bg_detail"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        for i, (col, val) in enumerate(zip(columns, row)):
            row_bg = C["bg_row_even"] if i % 2 == 0 else C["bg_row_odd"]
            rf = tk.Frame(inner, bg=row_bg)
            rf.pack(fill="x")
            if i > 0:
                tk.Frame(rf, bg=C["border_light"], height=1).pack(fill="x")
            ct = tk.Frame(rf, bg=row_bg, padx=16, pady=7)
            ct.pack(fill="x")

            kf = tk.Frame(ct, bg=row_bg, width=170)
            kf.pack(side="left", fill="y")
            kf.pack_propagate(False)
            tk.Label(kf, text=col, font=F["detail_key"],
                     bg=row_bg, fg=C["fg_header"],
                     anchor="ne", justify="right").pack(anchor="ne", pady=1)

            tk.Frame(ct, bg=C["border_light"], width=1).pack(side="left", fill="y", padx=12)

            display = str(val) if val is not None else "NULL"
            fg = C["fg_null"] if val is None else (
                C["fg_ts"] if col in TS_COLS else C["fg_primary"])

            vf = tk.Frame(ct, bg=row_bg)
            vf.pack(side="left", fill="both", expand=True)

            lines = display.count("\n") + 1
            if len(display) > 80 or lines > 1:
                hs = ttk.Scrollbar(vf, orient="horizontal")
                txt = tk.Text(vf, font=F["detail_val"], height=min(lines, 5),
                              wrap="none", bd=1, relief="solid",
                              bg=C["bg_input"], fg=fg,
                              highlightthickness=1,
                              highlightbackground=C["border_input"],
                              highlightcolor=C["border_focus"],
                              xscrollcommand=hs.set,
                              selectbackground=C["bg_row_sel"])
                hs.configure(command=txt.xview)
                txt.pack(fill="x", expand=True)
                hs.pack(fill="x")
                txt.insert("1.0", display)
                txt.configure(state="disabled")
            else:
                tk.Label(vf, text=display, font=F["detail_val"],
                         bg=row_bg, fg=fg,
                         anchor="w", justify="left",
                         wraplength=400).pack(anchor="w", fill="x", expand=True)

        bf = tk.Frame(self, bg=C["bg_detail"], pady=10)
        bf.pack(fill="x")
        tk.Button(bf, text="关  闭", font=F["normal"],
                  bg=C["bg_btn"], fg=C["fg_btn"],
                  bd=0, padx=20, pady=5,
                  activebackground=C["bg_btn_hover"],
                  activeforeground=C["fg_btn"],
                  cursor="hand2",
                  command=self.destroy).pack()


# ─────────────────────── 表内容视图 ───────────────────────
class TableView(tk.Frame):
    def __init__(self, parent, table_name):
        super().__init__(parent, bg=C["bg_content"])
        self.table_name = table_name
        self._columns = []
        self._rows = []
        self._filtered_rows = []
        self._search_var = tk.StringVar()
        self._search_placeholder = True
        self._build_ui()
        self._search_var.trace_add("write", self._on_search)
        
        self._load_data()

    def _build_ui(self):
        toolbar = tk.Frame(self, bg=C["bg_toolbar"], pady=9, padx=16)
        toolbar.pack(fill="x")

        left = tk.Frame(toolbar, bg=C["bg_toolbar"])
        left.pack(side="left", fill="y")
        self.lbl_table = tk.Label(left, text="", font=F["title"],
                                  bg=C["bg_toolbar"], fg=C["fg_primary"])
        self.lbl_table.pack(side="left", anchor="w")
        self.lbl_badge = tk.Label(left, text="", font=F["badge"],
                                  bg=C["bg_badge"], fg=C["fg_badge"],
                                  padx=6, pady=2)
        self.lbl_badge.pack(side="left", padx=8, anchor="w")

        right = tk.Frame(toolbar, bg=C["bg_toolbar"])
        right.pack(side="right", fill="y")

        sb = tk.Frame(right, bg=C["border_input"], padx=1, pady=1)
        sb.pack(side="left", padx=(0, 10))
        si = tk.Frame(sb, bg=C["bg_input"])
        si.pack()
        tk.Label(si, text="🔍", font=(KAI, 9),
                 bg=C["bg_input"], fg=C["fg_tertiary"], padx=5).pack(side="left")
        self.search_entry = tk.Entry(si, textvariable=self._search_var,
                                     font=F["normal"],
                                     bg=C["bg_input"], fg=C["fg_tertiary"],
                                     insertbackground=C["fg_primary"],
                                     relief="flat", bd=0, width=20)
        self.search_entry.insert(0, "搜索内容…")
        self.search_entry.pack(side="left", ipady=4, padx=(0, 8))
        self.search_entry.bind("<FocusIn>",  self._sin)
        self.search_entry.bind("<FocusOut>", self._sout)

        tk.Button(right, text="↺  刷新", font=F["normal"],
                  bg=C["bg_btn"], fg=C["fg_btn"],
                  bd=0, padx=12, pady=4,
                  activebackground=C["bg_btn_hover"],
                  activeforeground=C["fg_btn"],
                  cursor="hand2",
                  command=self._load_data).pack(side="left")

        tk.Frame(self, bg=C["border_light"], height=1).pack(fill="x")

        tree_wrap = tk.Frame(self, bg=C["bg_content"])
        tree_wrap.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(tree_wrap, show="headings", selectmode="browse")
        vs = ttk.Scrollbar(tree_wrap, orient="vertical",   command=self.tree.yview)
        hs = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self._on_dbl)
        self.tree.bind("<Return>",   self._on_dbl)

        tk.Frame(self, bg=C["border_light"], height=1).pack(fill="x", side="bottom")
        sbar = tk.Frame(self, bg=C["bg_toolbar"], pady=4, padx=16)
        sbar.pack(fill="x", side="bottom")
        self.lbl_status = tk.Label(sbar, text="", font=F["tiny"],
                                   bg=C["bg_toolbar"], fg=C["fg_tertiary"], anchor="w")
        self.lbl_status.pack(side="left")
        tk.Label(sbar, text="双击行或按 Enter 查看完整字段内容",
                 font=F["tiny"], bg=C["bg_toolbar"],
                 fg=C["fg_tertiary"], anchor="e").pack(side="right")

    def _sin(self, _e):
        if self._search_placeholder:
            self.search_entry.delete(0, "end")
            self.search_entry.configure(fg=C["fg_primary"])
            self._search_placeholder = False

    def _sout(self, _e):
        if not self.search_entry.get():
            self.search_entry.insert(0, "搜索内容…")
            self.search_entry.configure(fg=C["fg_tertiary"])
            self._search_placeholder = True

    def _on_search(self, *_):
        kw = self._search_var.get().strip().lower()
        if self._search_placeholder or not kw:
            self._filtered_rows = self._rows
        else:
            self._filtered_rows = [
                r for r in self._rows
                if any(kw in str(v).lower() for v in r)
            ]
        self._render_rows()

    def _load_data(self):
        try:
            self._columns = get_table_columns(self.table_name)
            self._rows, total = get_table_rows(self.table_name, limit=300)
        except Exception as e:
            messagebox.showerror("读取失败", f"无法读取表 {self.table_name}:\n{e}")
            return
        self._filtered_rows = self._rows
        shown = len(self._rows)
        self.lbl_table.configure(text=self.table_name)
        badge = f"共 {total} 条" + (f"  ·  最新 {shown} 条" if total > shown else "")
        self.lbl_badge.configure(text=badge)

        self.tree.configure(columns=self._columns)
        for col in self._columns:
            anchor = "e" if col.endswith(("_id", "_ts", "id")) else "w"
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            w = 160 if col in LONG_COLS else 120 if col in TS_COLS else 90
            self.tree.column(col, width=w, minwidth=50, anchor=anchor)

        self._render_rows()

    def _render_rows(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, row in enumerate(self._filtered_rows):
            values = [fmt_cell(c, v)[0] for c, v in zip(self._columns, row)]
            self.tree.insert("", "end", values=values,
                             tags=("odd" if i % 2 else "even",))
        self.lbl_status.configure(
            text=f"显示 {len(self._filtered_rows)} 条"
                 + (f"（共 {len(self._rows)} 条）"
                    if len(self._filtered_rows) != len(self._rows) else ""))

    def _sort_by(self, col):
        try:
            idx = self._columns.index(col)
        except ValueError:
            return
        try:
            self._filtered_rows.sort(key=lambda r: (r[idx] is None, r[idx]))
        except TypeError:
            self._filtered_rows.sort(key=lambda r: str(r[idx] or ""))
        self._render_rows()

    def _on_dbl(self, _e=None):
        item = self.tree.focus()
        if not item:
            return
        idx = self.tree.index(item)
        if idx < len(self._filtered_rows):
            RowDetailDialog(self, self._columns, self._filtered_rows[idx])


# ─────────────────────── 侧栏：树形分组 ───────────────────────
class TreeGroup(tk.Frame):
    """
    可折叠分组节点。
    展开时显示子叶节点（表），折叠时隐藏。
    点击组头行切换展开/折叠。
    """
    def __init__(self, parent, label, tables_counts,
                 on_leaf_click, expanded=True, **kw):
        super().__init__(parent, bg=C["bg_sidebar"], **kw)
        self._expanded = expanded
        self._on_leaf_click = on_leaf_click
        self._leaves = {}

        # ── 组头 ──
        self._hdr = tk.Frame(self, bg=C["bg_group_hdr"], cursor="hand2")
        self._hdr.pack(fill="x")

        self._arrow = tk.Label(self._hdr,
                               text="▾" if expanded else "▸",
                               font=(KAI, 9),
                               bg=C["bg_group_hdr"],
                               fg=C["fg_group"],
                               width=2)
        self._arrow.pack(side="left", padx=(10, 2), pady=5)

        self._grp_lbl = tk.Label(self._hdr, text=label,
                                 font=F["section"],
                                 bg=C["bg_group_hdr"],
                                 fg=C["fg_group"],
                                 anchor="w")
        self._grp_lbl.pack(side="left", fill="x", expand=True)

        cnt = tk.Label(self._hdr, text=str(len(tables_counts)),
                       font=F["badge"],
                       bg=C["bg_badge"], fg=C["fg_badge"],
                       padx=5, pady=1)
        cnt.pack(side="right", padx=8)

        for w in (self._hdr, self._arrow, self._grp_lbl, cnt):
            w.bind("<Button-1>", self._toggle)
            w.bind("<Enter>",    lambda e: self._hdr.configure(bg=C["bg_node_hover"]))
            w.bind("<Leave>",    lambda e: self._hdr.configure(bg=C["bg_group_hdr"]))

        tk.Frame(self, bg=C["border_light"], height=1).pack(fill="x")

        # ── 子项容器 ──
        self._body = tk.Frame(self, bg=C["bg_sidebar"])
        if expanded:
            self._body.pack(fill="x")

        for tname, row_cnt in tables_counts:
            leaf = TreeLeaf(self._body, tname, row_cnt,
                            on_click=on_leaf_click)
            leaf.pack(fill="x")
            self._leaves[tname] = leaf

    def _toggle(self, _e=None):
        self._expanded = not self._expanded
        self._arrow.configure(text="▾" if self._expanded else "▸")
        if self._expanded:
            self._body.pack(fill="x")
        else:
            self._body.pack_forget()

    def all_leaves(self):
        return dict(self._leaves)


class TreeLeaf(tk.Frame):
    """侧栏中的单张表叶节点"""
    def __init__(self, parent, table_name, row_count, on_click, **kw):
        super().__init__(parent, bg=C["bg_sidebar"], cursor="hand2", **kw)
        self.table_name = table_name
        self._selected  = False
        self._on_click  = on_click

        self._inner = tk.Frame(self, bg=C["bg_sidebar"])
        self._inner.pack(fill="x")

        # 缩进
        tk.Label(self._inner, text="", width=3,
                 bg=C["bg_sidebar"]).pack(side="left")

        # 表图标
        self._icon = tk.Label(self._inner, text="▤",
                              font=(KAI, 9),
                              bg=C["bg_sidebar"],
                              fg=C["fg_tertiary"])
        self._icon.pack(side="left", padx=(4, 5), pady=5)

        self._lbl = tk.Label(self._inner, text=table_name,
                             font=F["tag"],
                             bg=C["bg_sidebar"],
                             fg=C["fg_primary"],
                             anchor="w")
        self._lbl.pack(side="left", fill="x", expand=True)

        self._badge = tk.Label(self._inner,
                               text=str(row_count),
                               font=F["badge"],
                               bg=C["bg_badge"], fg=C["fg_badge"],
                               padx=5, pady=1)
        self._badge.pack(side="right", padx=8)

        all_w = [self, self._inner, self._icon, self._lbl, self._badge]
        for w in all_w:
            w.bind("<Button-1>", lambda e: self._on_click(self.table_name))
            w.bind("<Enter>",    self._hover_on)
            w.bind("<Leave>",    self._hover_off)

    def set_selected(self, sel: bool):
        self._selected = sel
        bg  = C["bg_node_sel"]  if sel else C["bg_sidebar"]
        fg  = C["fg_node_sel"]  if sel else C["fg_primary"]
        ifg = C["fg_node_sel"]  if sel else C["fg_tertiary"]
        bbg = C["accent"]       if sel else C["bg_badge"]
        bfg = "#FFFFFF"         if sel else C["fg_badge"]
        for w in (self, self._inner):
            w.configure(bg=bg)
        self._lbl.configure(bg=bg, fg=fg)
        self._icon.configure(bg=bg, fg=ifg)
        self._badge.configure(bg=bbg, fg=bfg)

    def _hover_on(self, _e):
        if not self._selected:
            for w in (self, self._inner, self._lbl, self._icon, self._badge):
                w.configure(bg=C["bg_node_hover"])

    def _hover_off(self, _e):
        if not self._selected:
            self.set_selected(False)


# ─────────────────────── 主窗口 ───────────────────────
class DBBrowser(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"CoSearch DB Browser  —  {settings.db_name}")
        self.geometry("1280x780")
        self.minsize(900, 560)
        self.configure(bg=C["bg_app"])
        self._selected_table = None
        self._current_view   = None
        self._groups         = {}
        self._apply_ttk_style()
        self._build_ui()
        self._load_tables()

    def _apply_ttk_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Treeview",
                    font=F["normal"],
                    rowheight=28,
                    background=C["bg_content"],
                    fieldbackground=C["bg_content"],
                    foreground=C["fg_primary"],
                    borderwidth=0, relief="flat")
        s.configure("Treeview.Heading",
                    font=F["header"],
                    background=C["bg_toolbar"],
                    foreground=C["fg_header"],
                    relief="flat", padding=(8, 5))
        s.map("Treeview",
              background=[("selected", C["bg_row_sel"])],
              foreground=[("selected", C["fg_primary"])])
        s.map("Treeview.Heading",
              background=[("active", C["bg_badge"])])
        for o in ("Vertical", "Horizontal"):
            s.configure(f"{o}.TScrollbar",
                        troughcolor=C["bg_toolbar"],
                        background=C["border_mid"],
                        arrowsize=11)
            s.map(f"{o}.TScrollbar",
                  background=[("active", C["fg_tertiary"])])

    def _build_ui(self):
        # 顶栏
        topbar = tk.Frame(self, bg=C["bg_topbar"], height=50)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        logo = tk.Frame(topbar, bg=C["bg_topbar"])
        logo.pack(side="left", padx=18, fill="y")
        tk.Label(logo, text="⬡", font=(KAI, 15),
                 bg=C["bg_topbar"], fg=C["accent"]).pack(side="left", padx=(0, 7))
        tk.Label(logo, text="CoSearch", font=F["topbar"],
                 bg=C["bg_topbar"], fg=C["fg_topbar"]).pack(side="left")
        tk.Label(logo, text="DB Browser", font=F["topbar_sub"],
                 bg=C["bg_topbar"], fg=C["fg_topbar_sub"]).pack(side="left", padx=(6, 0))

        info = tk.Frame(topbar, bg=C["bg_topbar"])
        info.pack(side="right", padx=18, fill="y")
        self.lbl_db = tk.Label(info, text=settings.db_name,
                               font=F["normal"], bg=C["bg_topbar"],
                               fg=C["fg_topbar"])
        self.lbl_db.pack(side="right")
        tk.Label(info, text=f"{settings.db_host}:{settings.db_port}  ·",
                 font=F["topbar_sub"], bg=C["bg_topbar"],
                 fg=C["fg_topbar_sub"]).pack(side="right", padx=(0, 6))

        # 主体
        body = tk.Frame(self, bg=C["bg_app"])
        body.pack(fill="both", expand=True)

        # 侧栏外框
        sidebar_outer = tk.Frame(body, bg=C["bg_sidebar"], width=224)
        sidebar_outer.pack(side="left", fill="y")
        sidebar_outer.pack_propagate(False)

        # 侧栏标题
        sh = tk.Frame(sidebar_outer, bg=C["bg_group_hdr"], pady=7, padx=14)
        sh.pack(fill="x")
        tk.Label(sh, text="数据库", font=F["section"],
                 bg=C["bg_group_hdr"], fg=C["fg_group"]).pack(side="left")
        self.lbl_tbl_count = tk.Label(sh, text="", font=F["badge"],
                                      bg=C["bg_badge"], fg=C["fg_badge"],
                                      padx=5, pady=1)
        self.lbl_tbl_count.pack(side="right")
        tk.Frame(sidebar_outer, bg=C["border_light"], height=1).pack(fill="x")

        # 可滚动树
        lo = tk.Frame(sidebar_outer, bg=C["bg_sidebar"])
        lo.pack(fill="both", expand=True)

        self.sc = tk.Canvas(lo, bg=C["bg_sidebar"],
                            highlightthickness=0, bd=0, yscrollincrement=2)
        sv = ttk.Scrollbar(lo, orient="vertical", command=self.sc.yview)
        self.sc.configure(yscrollcommand=sv.set)
        self.sc.pack(side="left", fill="both", expand=True)
        sv.pack(side="right", fill="y")

        self.tree_inner = tk.Frame(self.sc, bg=C["bg_sidebar"])
        self.tree_inner.bind("<Configure>",
            lambda e: self.sc.configure(scrollregion=self.sc.bbox("all")))
        self._sc_win = self.sc.create_window((0, 0), window=self.tree_inner, anchor="nw")
        self.sc.bind("<Configure>",
            lambda e: self.sc.itemconfig(self._sc_win, width=e.width))
        self.sc.bind("<MouseWheel>",
            lambda e: self.sc.yview_scroll(int(-1*(e.delta/120)), "units"))

        # 侧栏底部
        tk.Frame(sidebar_outer, bg=C["border_light"], height=1).pack(fill="x", side="bottom")
        sf = tk.Frame(sidebar_outer, bg=C["bg_group_hdr"], pady=6, padx=14)
        sf.pack(fill="x", side="bottom")
        tk.Button(sf, text="↺  刷新列表", font=F["small"],
                  bg=C["bg_group_hdr"], fg=C["fg_link"],
                  bd=0, relief="flat",
                  activebackground=C["accent_light"],
                  activeforeground=C["accent"],
                  cursor="hand2",
                  command=self._load_tables).pack(fill="x")

        tk.Frame(body, bg=C["border_light"], width=1).pack(side="left", fill="y")

        # 内容区
        self.content_area = tk.Frame(body, bg=C["bg_content"])
        self.content_area.pack(side="left", fill="both", expand=True)

        self.welcome = tk.Frame(self.content_area, bg=C["bg_content"])
        self.welcome.pack(fill="both", expand=True)
        center = tk.Frame(self.welcome, bg=C["bg_content"])
        center.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(center, text="⬡", font=(KAI, 46),
                 bg=C["bg_content"], fg=C["border_mid"]).pack(pady=(0, 10))
        tk.Label(center, text="请从左侧选择一张表",
                 font=(KAI, 13), bg=C["bg_content"],
                 fg=C["fg_tertiary"]).pack()
        tk.Label(center, text="支持搜索过滤  ·  点击列头排序  ·  双击行查看完整内容",
                 font=F["tiny"], bg=C["bg_content"],
                 fg=C["fg_tertiary"]).pack(pady=(5, 0))

    def _load_tables(self):
        try:
            tables = get_all_tables()
        except Exception as e:
            messagebox.showerror("连接失败", f"无法连接数据库:\n{e}")
            return

        for w in self.tree_inner.winfo_children():
            w.destroy()
        self._groups.clear()

        base = [t for t in tables if t in ("channel_info", "user_info", "user_profile")]
        rest = sorted([t for t in tables if t not in base])

        def _make_group(label, tlist, expanded=True):
            if not tlist:
                return
            tc = []
            for t in tlist:
                try:
                    cnt = get_table_row_count(t)
                except Exception:
                    cnt = "?"
                tc.append((t, cnt))
            grp = TreeGroup(self.tree_inner, label, tc,
                            on_leaf_click=self._on_table_click,
                            expanded=expanded)
            grp.pack(fill="x")
            self._groups[label] = grp

        if base:
            _make_group("基础表", base, expanded=True)
        _make_group("业务表", rest, expanded=True)

        self.lbl_tbl_count.configure(text=str(len(tables)))
        self.lbl_db.configure(text=settings.db_name)

    def _all_leaves(self):
        result = {}
        for grp in self._groups.values():
            result.update(grp.all_leaves())
        return result

    def _on_table_click(self, table_name: str):
        if table_name == self._selected_table:
            return
        leaves = self._all_leaves()
        if self._selected_table and self._selected_table in leaves:
            leaves[self._selected_table].set_selected(False)
        self._selected_table = table_name
        if table_name in leaves:
            leaves[table_name].set_selected(True)
        self._show_table(table_name)

    def _show_table(self, table_name: str):
        if self.welcome.winfo_ismapped():
            self.welcome.pack_forget()
        if self._current_view is not None:
            self._current_view.destroy()
        self._current_view = TableView(self.content_area, table_name)
        self._current_view.pack(fill="both", expand=True)
        self.title(f"{table_name}  —  CoSearch DB Browser")

        tree = self._current_view.tree
        tree.tag_configure("even", background=C["bg_row_even"])
        tree.tag_configure("odd",  background=C["bg_row_odd"])


# ─────────────────────── 程序入口 ───────────────────────
if __name__ == "__main__":
    app = DBBrowser()
    app.mainloop()