import { useState } from "react";

// ─── 配置：修改为你的实际后端地址 ───────────────────────────────────────────
// 启动方式见文件末尾的 db_server.py
const API = "http://localhost:7788";

// ─── 颜色 / 样式常量 ─────────────────────────────────────────────────────────
const COLORS = {
  bg: "#0f1117",
  card: "#1a1d27",
  border: "#2a2d3a",
  accent: "#6c63ff",
  accent2: "#00d4aa",
  text: "#e2e8f0",
  muted: "#64748b",
  tag: "#1e3a5f",
  tagText: "#7dd3fc",
  danger: "#ef4444",
  warn: "#f59e0b",
  success: "#10b981",
};

const TABS = [
  { id: "profiles",  label: "👤 用户画像",   icon: "👤" },
  { id: "channels",  label: "💬 对话记录",   icon: "💬" },
  { id: "users",     label: "🗂️ 用户映射",   icon: "🗂️" },
];

// ─── 工具函数 ─────────────────────────────────────────────────────────────────
function Tag({ text, color = COLORS.tag, textColor = COLORS.tagText }) {
  return (
    <span style={{
      background: color, color: textColor,
      borderRadius: 4, padding: "2px 8px", fontSize: 12,
      marginRight: 4, marginBottom: 4, display: "inline-block",
    }}>
      {text}
    </span>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + "22", color: color,
      border: `1px solid ${color}44`,
      borderRadius: 12, padding: "1px 8px", fontSize: 11, fontWeight: 600,
    }}>
      {text}
    </span>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: COLORS.card, border: `1px solid ${COLORS.border}`,
      borderRadius: 10, padding: 16, marginBottom: 12, ...style,
    }}>
      {children}
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ textAlign: "center", padding: 40, color: COLORS.muted }}>
      <div style={{
        display: "inline-block", width: 32, height: 32,
        border: `3px solid ${COLORS.border}`,
        borderTop: `3px solid ${COLORS.accent}`,
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <div style={{ marginTop: 8 }}>加载中...</div>
    </div>
  );
}

function EmptyState({ msg = "暂无数据" }) {
  return (
    <div style={{ textAlign: "center", padding: 60, color: COLORS.muted }}>
      <div style={{ fontSize: 40 }}>🗃️</div>
      <div style={{ marginTop: 8 }}>{msg}</div>
    </div>
  );
}

// ─── 用户画像面板 ─────────────────────────────────────────────────────────────
function ProfilesPanel() {
  const [profiles, setProfiles] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);

  async function load() {
    setLoading(true); setError("");
    try {
      const r = await fetch(`${API}/profiles`);
      const d = await r.json();
      setProfiles(d.profiles || []);
    } catch (e) { setError("连接失败：" + e.message); }
    setLoading(false);
  }

  if (!profiles && !loading) return (
    <div style={{ padding: 24 }}>
      <button onClick={load} style={btnStyle(COLORS.accent)}>
        📥 加载用户画像
      </button>
      {error && <div style={{ color: COLORS.danger, marginTop: 12 }}>{error}</div>}
    </div>
  );

  if (loading) return <Spinner />;
  if (!profiles?.length) return <EmptyState msg="user_profile 表为空" />;

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <span style={{ color: COLORS.muted }}>共 {profiles.length} 条画像</span>
        <button onClick={load} style={btnStyle(COLORS.muted, true)}>🔄 刷新</button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 12 }}>
        {profiles.map((p, i) => (
          <Card key={i} style={{
            cursor: "pointer",
            border: selected === i ? `1px solid ${COLORS.accent}` : `1px solid ${COLORS.border}`,
            transition: "border 0.2s",
          }} >
            <div onClick={() => setSelected(selected === i ? null : i)}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: COLORS.text, fontSize: 15 }}>
                  👤 {p.user_name || p.user_id}
                </span>
                <Badge text={p.major || "专业未知"} color={COLORS.accent} />
              </div>
              <div style={{ color: COLORS.muted, fontSize: 12, marginBottom: 8 }}>
                ID: {p.user_id} · 更新: {p.updated_at || "—"}
              </div>

              {p.research_interests?.length > 0 && (
                <div style={{ marginBottom: 6 }}>
                  <div style={{ color: COLORS.muted, fontSize: 11, marginBottom: 3 }}>研究兴趣</div>
                  {p.research_interests.map((t, j) => <Tag key={j} text={t} />)}
                </div>
              )}
              {p.methodology?.length > 0 && (
                <div style={{ marginBottom: 6 }}>
                  <div style={{ color: COLORS.muted, fontSize: 11, marginBottom: 3 }}>擅长方法</div>
                  {p.methodology.map((t, j) => <Tag key={j} text={t} color="#1a3a2a" textColor="#6ee7b7" />)}
                </div>
              )}
              {selected === i && p.keywords?.length > 0 && (
                <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px solid ${COLORS.border}` }}>
                  <div style={{ color: COLORS.muted, fontSize: 11, marginBottom: 3 }}>关键词</div>
                  {p.keywords.map((t, j) => <Tag key={j} text={t} color="#2a1a3a" textColor="#c4b5fd" />)}
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ─── 对话记录面板 ─────────────────────────────────────────────────────────────
function ChannelsPanel() {
  const [channels, setChannels] = useState(null);
  const [selected, setSelected] = useState("");
  const [convs, setConvs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [showBot, setShowBot] = useState(true);

  async function loadChannels() {
    setLoading(true); setError("");
    try {
      const r = await fetch(`${API}/channels`);
      const d = await r.json();
      setChannels(d.channels || []);
    } catch (e) { setError("连接失败：" + e.message); }
    setLoading(false);
  }

  async function loadConvs(ch) {
    setSelected(ch); setConvs(null); setLoading(true);
    try {
      const r = await fetch(`${API}/convs/${encodeURIComponent(ch)}`);
      const d = await r.json();
      setConvs(d.rows || []);
    } catch (e) { setError("加载失败：" + e.message); }
    setLoading(false);
  }

  const filteredConvs = convs?.filter(row => {
    if (!showBot && row.speaker === "CoSearchAgent") return false;
    if (filter && !row.utterance?.includes(filter) && !row.speaker?.includes(filter)) return false;
    return true;
  });

  if (!channels && !loading) return (
    <div style={{ padding: 24 }}>
      <button onClick={loadChannels} style={btnStyle(COLORS.accent)}>
        📥 加载频道列表
      </button>
      {error && <div style={{ color: COLORS.danger, marginTop: 12 }}>{error}</div>}
    </div>
  );

  if (loading && !channels) return <Spinner />;

  return (
    <div style={{ display: "flex", height: "calc(100vh - 140px)", gap: 0 }}>
      {/* 左侧频道列表 */}
      <div style={{
        width: 200, background: COLORS.bg, borderRight: `1px solid ${COLORS.border}`,
        overflow: "auto", flexShrink: 0,
      }}>
        <div style={{ padding: "12px 8px", color: COLORS.muted, fontSize: 11, fontWeight: 600 }}>
          CHANNELS
        </div>
        {channels?.map((ch, i) => (
          <div key={i} onClick={() => loadConvs(ch.name)}
            style={{
              padding: "8px 12px", cursor: "pointer", fontSize: 13,
              background: selected === ch.name ? COLORS.accent + "22" : "transparent",
              color: selected === ch.name ? COLORS.accent : COLORS.text,
              borderLeft: selected === ch.name ? `2px solid ${COLORS.accent}` : "2px solid transparent",
              transition: "all 0.15s",
            }}>
            # {ch.display_name || ch.name}
            <span style={{ color: COLORS.muted, fontSize: 11, marginLeft: 4 }}>
              {ch.count}
            </span>
          </div>
        ))}
      </div>

      {/* 右侧对话内容 */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {!selected ? (
          <div style={{ padding: 40, color: COLORS.muted, textAlign: "center" }}>
            ← 选择一个频道查看对话记录
          </div>
        ) : (
          <>
            <div style={{
              padding: "10px 16px", borderBottom: `1px solid ${COLORS.border}`,
              display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap",
            }}>
              <input
                placeholder="搜索说话人或内容..."
                value={filter}
                onChange={e => setFilter(e.target.value)}
                style={inputStyle()}
              />
              <label style={{ color: COLORS.muted, fontSize: 13, cursor: "pointer", userSelect: "none" }}>
                <input type="checkbox" checked={showBot} onChange={e => setShowBot(e.target.checked)}
                  style={{ marginRight: 4 }} />
                显示Bot回复
              </label>
              <span style={{ color: COLORS.muted, fontSize: 12, marginLeft: "auto" }}>
                {filteredConvs?.length ?? "—"} 条
              </span>
            </div>
            <div style={{ flex: 1, overflow: "auto", padding: 12 }}>
              {loading ? <Spinner /> : filteredConvs?.length === 0 ? <EmptyState /> : (
                filteredConvs?.map((row, i) => (
                  <ConvRow key={i} row={row} />
                ))
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ConvRow({ row }) {
  const [expanded, setExpanded] = useState(false);
  const isBot = row.speaker === "CoSearchAgent";
  const speakerColor = isBot ? COLORS.accent2 : COLORS.accent;

  return (
    <div style={{
      marginBottom: 8, padding: "10px 14px",
      background: isBot ? COLORS.card : COLORS.bg,
      border: `1px solid ${isBot ? COLORS.accent2 + "33" : COLORS.border}`,
      borderRadius: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: speakerColor, fontWeight: 600, fontSize: 13 }}>
          {isBot ? "🤖" : "👤"} {row.speaker}
        </span>
        <span style={{ color: COLORS.muted, fontSize: 11 }}>{row.timestamp}</span>
      </div>
      <div style={{ color: COLORS.text, fontSize: 13, lineHeight: 1.5, whiteSpace: "pre-wrap" }}>
        {row.utterance}
      </div>
      {(row.query || row.search_results) && (
        <div style={{ marginTop: 6 }}>
          <button onClick={() => setExpanded(!expanded)}
            style={{ background: "none", border: "none", color: COLORS.muted,
              cursor: "pointer", fontSize: 11, padding: 0 }}>
            {expanded ? "▲ 收起" : "▼ 展开元数据"}
          </button>
          {expanded && (
            <div style={{
              marginTop: 8, padding: 10, background: COLORS.bg,
              borderRadius: 6, fontSize: 11, color: COLORS.muted,
              border: `1px solid ${COLORS.border}`,
            }}>
              {row.query && <div><b>Query:</b> {row.query}</div>}
              {row.rewrite_query && <div><b>重写:</b> {row.rewrite_query}</div>}
              {row.clarify && <div><b>澄清:</b> {row.clarify}</div>}
              {row.infer_time && <div><b>耗时:</b> {row.infer_time}</div>}
              {row.search_results && (
                <div><b>搜索结果:</b> {row.search_results.slice(0, 200)}...</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── 用户映射面板 ─────────────────────────────────────────────────────────────
function UsersPanel() {
  const [users, setUsers] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function load() {
    setLoading(true); setError("");
    try {
      const r = await fetch(`${API}/users`);
      const d = await r.json();
      setUsers(d.users || []);
    } catch (e) { setError("连接失败：" + e.message); }
    setLoading(false);
  }

  if (!users && !loading) return (
    <div style={{ padding: 24 }}>
      <button onClick={load} style={btnStyle(COLORS.accent)}>📥 加载用户映射</button>
      {error && <div style={{ color: COLORS.danger, marginTop: 12 }}>{error}</div>}
    </div>
  );
  if (loading) return <Spinner />;
  if (!users?.length) return <EmptyState msg="user_info 表为空" />;

  return (
    <div style={{ padding: 16 }}>
      <div style={{ color: COLORS.muted, marginBottom: 12 }}>共 {users.length} 条</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${COLORS.border}` }}>
            {["user_id", "user_name"].map(h => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left",
                color: COLORS.muted, fontWeight: 600 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.map((u, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}22`,
              background: i % 2 === 0 ? "transparent" : COLORS.card + "88" }}>
              <td style={{ padding: "8px 12px", color: COLORS.muted, fontFamily: "monospace" }}>
                {u.user_id}
              </td>
              <td style={{ padding: "8px 12px", color: COLORS.text, fontWeight: 500 }}>
                {u.user_name}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── 样式工具 ─────────────────────────────────────────────────────────────────
function btnStyle(color, small = false) {
  return {
    background: color + "22", color, border: `1px solid ${color}55`,
    borderRadius: 6, padding: small ? "4px 10px" : "8px 18px",
    cursor: "pointer", fontSize: small ? 12 : 14, fontWeight: 500,
    transition: "background 0.2s",
  };
}
function inputStyle() {
  return {
    background: COLORS.bg, border: `1px solid ${COLORS.border}`,
    borderRadius: 6, padding: "5px 10px", color: COLORS.text,
    fontSize: 13, outline: "none", minWidth: 200,
  };
}

// ─── 主应用 ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("profiles");

  return (
    <div style={{
      background: COLORS.bg, minHeight: "100vh",
      color: COLORS.text, fontFamily: "'Inter', 'PingFang SC', system-ui, sans-serif",
    }}>
      {/* 顶栏 */}
      <div style={{
        background: COLORS.card, borderBottom: `1px solid ${COLORS.border}`,
        padding: "0 20px", display: "flex", alignItems: "center", gap: 24,
        height: 50,
      }}>
        <span style={{ fontWeight: 700, color: COLORS.accent, fontSize: 15 }}>
          🔍 CoSearch Memory Viewer
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: tab === t.id ? COLORS.accent + "22" : "transparent",
              color: tab === t.id ? COLORS.accent : COLORS.muted,
              border: "none", borderRadius: 6, padding: "6px 14px",
              cursor: "pointer", fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
              transition: "all 0.15s",
            }}>
              {t.label}
            </button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", fontSize: 11, color: COLORS.muted }}>
          后端: {API}
        </div>
      </div>

      {/* 内容区 */}
      <div style={{ overflow: "auto" }}>
        {tab === "profiles"  && <ProfilesPanel />}
        {tab === "channels"  && <ChannelsPanel />}
        {tab === "users"     && <UsersPanel />}
      </div>
    </div>
  );
}