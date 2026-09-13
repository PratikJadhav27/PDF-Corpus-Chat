"""
app/styles.py

Single source of truth for all injected CSS.
Call get_css() once in app/main.py via st.markdown(get_css(), unsafe_allow_html=True).

Design system: DocMind Dark — Linear/Vercel/Raycast aesthetic
  Background : #0A0D14 (deep navy)
  Surface    : #111827 (dark card)
  Border     : #1F2937 (subtle)
  Primary    : #3B82F6 (electric blue)
  Success    : #10B981 (emerald)
  Warning    : #F59E0B (amber)
  Danger     : #F43F5E (rose)
  Violet     : #8B5CF6
  Text-1     : #F1F5F9 (primary)
  Text-2     : #94A3B8 (secondary)
  Text-3     : #475569 (muted)

IMPORTANT: Do NOT use `[class*="st-"]` or broad universal selectors — they
break Streamlit's Material Symbols icon font (causes "keyboard_double_arrow_right"
text to appear instead of icon glyphs). Target specific data-testid attributes.
"""


def get_css() -> str:
    return """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   GOOGLE FONTS
   ═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   CSS VARIABLES
   ═══════════════════════════════════════════════════════════════════════════ */
:root {
    --bg-base:      #0A0D14;
    --bg-surface:   #111827;
    --bg-elevated:  #1A2235;
    --border:       #1F2937;
    --border-light: #374151;
    --primary:      #3B82F6;
    --primary-dark: #1D4ED8;
    --primary-glow: rgba(59,130,246,0.15);
    --success:      #10B981;
    --warning:      #F59E0B;
    --danger:       #F43F5E;
    --violet:       #8B5CF6;
    --text-1:       #F1F5F9;
    --text-2:       #94A3B8;
    --text-3:       #475569;
    --font-body:    'Inter', -apple-system, sans-serif;
    --font-display: 'Space Grotesk', -apple-system, sans-serif;
    --font-mono:    'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm:    6px;
    --radius:       10px;
    --radius-lg:    14px;
    --radius-full:  9999px;
    --transition:   all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP BACKGROUND (targeted, NOT universal)
   ═══════════════════════════════════════════════════════════════════════════ */
.stApp {
    background-color: var(--bg-base) !important;
}

/* Main content area text — only body-level, not icon buttons */
.stApp p, .stApp li, .stApp label {
    color: var(--text-1);
    font-family: var(--font-body);
}


/* Hide deploy button and Streamlit menu — but NOT the header itself (it has the sidebar toggle) */
#stDecoration { display: none !important; }
[data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stMainMenuButton"] { display: none !important; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }

/* Style the top Streamlit header bar to match dark theme */
[data-testid="stHeader"] {
    background-color: var(--bg-base) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* Make the sidebar collapse/expand button clearly visible */
[data-testid="collapsedControl"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 0 var(--radius) var(--radius) 0 !important;
    color: var(--text-2) !important;
}
[data-testid="collapsedControl"]:hover {
    background-color: var(--bg-elevated) !important;
    border-color: var(--primary) !important;
    color: var(--text-1) !important;
}

/* Block container padding — compensate for visible header */
.stApp .block-container {
    padding-top: 1rem !important;
    padding-bottom: 5rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CUSTOM SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #0D111B !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .sidebar-label {
    font-family: var(--font-body) !important;
}
/* IMPORTANT: do NOT set font-family on sidebar spans — they include icon glyphs
   that rely on Material Symbols font (keyboard_double_arrow_right, etc.) */

/* Sidebar collapse button — keep it but fix background so it's not jarring */
[data-testid="collapsedControl"] {
    background-color: #0D111B !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 var(--radius) var(--radius) 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TABS (PILL STYLE)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-full) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
    width: fit-content !important;
    margin-bottom: 1.25rem !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-full) !important;
    padding: 6px 20px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    border: none !important;
    transition: var(--transition) !important;
    font-family: var(--font-body) !important;
}
[data-baseweb="tab"]:hover { color: var(--text-1) !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--primary) !important;
    color: #fff !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.35) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   BUTTONS — only style stButton, not icon buttons
   ═══════════════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: var(--bg-elevated) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: var(--transition) !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: var(--text-3) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2563EB !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.4) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════════════════════════════════════════ */
.stTextInput input,
.stTextArea textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}
.stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-1) !important;
}
[data-baseweb="select"] { background: var(--bg-elevated) !important; }
[data-baseweb="select"] * { color: var(--text-1) !important; }
[data-baseweb="popover"] { background: var(--bg-surface) !important; border: 1px solid var(--border-light) !important; }
[role="option"] { background: var(--bg-surface) !important; color: var(--text-1) !important; }
[role="option"]:hover { background: var(--bg-elevated) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   SLIDER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-baseweb="slider"] [role="slider"] { background: var(--primary) !important; }
[data-baseweb="slider"] div[data-testid="stSliderThumb"] { background: var(--primary) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════════════════════════════════ */
/* User message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    background: transparent !important;
    padding: 0.5rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--primary-dark) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.75rem 1.1rem !important;
    max-width: 72% !important;
    margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: #fff !important;
    margin: 0 !important;
}

/* AI message */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--primary) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    margin: 0.5rem 0 !important;
    animation: fadeUp 0.3s ease forwards !important;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-full) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stChatInputSubmitButton"] > button {
    background: var(--primary) !important;
    border-radius: var(--radius) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   st.metric
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetric"]:hover { border-color: var(--border-light) !important; }
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--text-1) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-3) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: var(--text-2) !important;
    /* NOTE: do NOT set font-family here — the icon glyph (arrow_drop_down) lives
       inside a span within summary and uses Material Symbols Rounded font.
       Overriding font-family breaks it and shows raw text. */
}

/* ═══════════════════════════════════════════════════════════════════════════
   MATERIAL SYMBOLS ICON PROTECTION
   Belt-and-suspenders: explicitly restore Material Symbols font for all
   Streamlit icon glyphs, regardless of what parent rules may have set.
   This prevents 'keyboard_double_arrow_right', 'arrow_drop_down' etc.
   from showing as raw text.
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="collapsedControl"] span,
[data-testid="stExpander"] summary span,
[data-testid="baseButton-header"] span,
[data-testid="stSidebar"] button span,
button[kind="header"] span {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS / SPINNER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ALERTS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stAlertContainer"] {
    border-radius: var(--radius-lg) !important;
}
[data-testid="stInfo"] {
    background: rgba(59,130,246,0.08) !important;
    border-color: rgba(59,130,246,0.25) !important;
}
[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.08) !important;
    border-color: rgba(16,185,129,0.25) !important;
}
[data-testid="stError"] {
    background: rgba(244,63,94,0.08) !important;
    border-color: rgba(244,63,94,0.25) !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.08) !important;
    border-color: rgba(245,158,11,0.25) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: rgba(59,130,246,0.03) !important;
    border: 1.5px dashed var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: var(--primary-glow) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {
    background: var(--primary) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: 99px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATAFRAME
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DIVIDERS
   ═══════════════════════════════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   DOCMIND CUSTOM COMPONENTS (injected via unsafe_allow_html)
   ═══════════════════════════════════════════════════════════════════════════ */

/* App header */
.docmind-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.docmind-header-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 20px rgba(59,130,246,0.3);
    flex-shrink: 0;
}
.docmind-header-text h1 {
    font-family: var(--font-display) !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    color: var(--text-1) !important;
}
.docmind-header-text .subtitle {
    font-size: 0.7rem;
    color: var(--text-3);
    font-family: var(--font-mono);
    letter-spacing: 0.04em;
}

/* Collection badge */
.collection-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #6EE7B7;
    font-family: var(--font-mono);
    margin-bottom: 0.75rem;
}
.collection-badge .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px #10B981;
    animation: blink 2s infinite;
    flex-shrink: 0;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* Sidebar section labels */
.sidebar-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 1rem 0 0.35rem 0;
    font-family: var(--font-body);
}

/* Metric cards */
.dm-metric-card {
    background: rgba(17,24,39,0.9);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.dm-metric-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.dm-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
}
.dm-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.dm-metric-label {
    font-size: 0.7rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Inter', sans-serif;
}
.dm-trend {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: 99px;
    margin-top: 6px;
}
.dm-trend-up    { background: rgba(16,185,129,0.12); color: #6EE7B7; }
.dm-trend-stable{ background: rgba(139,92,246,0.12); color: #C4B5FD; }
.dm-trend-down  { background: rgba(245,158,11,0.12); color: #FCD34D; }

/* Source strip */
.dm-source-strip {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 0.5rem 0 0.75rem;
    -webkit-overflow-scrolling: touch;
}
.dm-source-card {
    flex-shrink: 0;
    width: 230px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1rem;
    transition: var(--transition);
}
.dm-source-card:hover {
    border-color: var(--primary);
    box-shadow: 0 0 12px rgba(59,130,246,0.15);
}
.dm-sc-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
    font-family: 'Inter', sans-serif;
}
.dm-sc-meta {
    font-size: 0.67rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-3);
    margin-bottom: 6px;
}
.dm-sc-excerpt {
    font-size: 0.72rem;
    color: var(--text-2);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font-family: 'Inter', sans-serif;
}

/* Latency badges */
.dm-latency-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.dm-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
}
.dm-badge-blue   { background: rgba(59,130,246,0.1);  border-color: rgba(59,130,246,0.3);  color: #93C5FD; }
.dm-badge-violet { background: rgba(139,92,246,0.1);  border-color: rgba(139,92,246,0.3);  color: #C4B5FD; }
.dm-badge-amber  { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.3);  color: #FCD34D; }
.dm-badge-green  { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.3);  color: #6EE7B7; }

/* Index status bar */
.dm-status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 99px;
    width: fit-content;
    margin-bottom: 1rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #6EE7B7;
}
.dm-status-dot-inner {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 6px #10B981;
    animation: blink 2s infinite;
}

/* Section sub-headers */
.dm-section {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-1);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    margin: 1.25rem 0 0.75rem;
}

/* Empty state */
.dm-empty {
    text-align: center;
    padding: 3rem 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    color: var(--text-3);
    font-size: 0.85rem;
    font-family: 'Inter', sans-serif;
}
.dm-empty .icon { font-size: 2.5rem; opacity: 0.3; margin-bottom: 0.75rem; }

/* Benchmark row */
.dm-bench-row {
    margin-bottom: 1rem;
}
.dm-bench-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
    font-size: 0.8rem;
    font-family: 'Inter', sans-serif;
    color: var(--text-1);
}
.dm-bench-label .score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-3);
}
.dm-bench-track {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.dm-bench-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* Analytics page header */
.dm-page-header {
    margin-bottom: 1.25rem;
}
.dm-page-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-1);
    margin: 0 0 2px;
}
.dm-page-header p {
    font-size: 0.78rem;
    color: var(--text-3);
    margin: 0;
    font-family: 'Inter', sans-serif;
}

/* Citation pills (inside chat messages) */
.dm-citation {
    display: inline-flex;
    align-items: center;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 5px;
    padding: 1px 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #93C5FD;
    margin: 0 3px;
}
</style>
"""
