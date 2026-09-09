"""
app/styles.py

Single source of truth for all injected CSS.
Returns a <style> block string that is injected via st.markdown(unsafe_allow_html=True).

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
"""


def get_css() -> str:
    return """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   GOOGLE FONTS
   ═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   CSS VARIABLES (DESIGN TOKENS)
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
   GLOBAL RESETS
   ═══════════════════════════════════════════════════════════════════════════ */
html, body, [class*="st-"] {
    font-family: var(--font-body) !important;
    color: var(--text-1) !important;
}

/* Main app background */
.stApp {
    background-color: var(--bg-base) !important;
}

/* Remove Streamlit's default header padding */
.stApp > header { display: none !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 100% !important;
}

/* Hide Streamlit menu & footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ═══════════════════════════════════════════════════════════════════════════
   CUSTOM SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ═══════════════════════════════════════════════════════════════════════════
   APP HEADER
   ═══════════════════════════════════════════════════════════════════════════ */
.docmind-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 1.25rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.docmind-header-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, var(--primary), var(--violet));
    border-radius: var(--radius);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 20px var(--primary-glow);
    flex-shrink: 0;
}
.docmind-header-text h1 {
    font-family: var(--font-display) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text-1) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    background: linear-gradient(135deg, #fff 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.docmind-header-text span {
    font-size: 0.72rem;
    color: var(--text-3);
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #0D111B !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem !important; }

/* Sidebar section labels */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 1rem 0 0.4rem 0;
}

/* Collection badge */
.collection-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: var(--radius-full);
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--success);
    font-family: var(--font-mono);
    margin-bottom: 0.75rem;
    width: 100%;
}
.collection-badge::before {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Upload dropzone */
[data-testid="stFileUploader"] {
    background: rgba(59,130,246,0.03) !important;
    border: 1.5px dashed var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: var(--primary-glow) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-2) !important; }

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
    margin-bottom: 1.5rem !important;
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
    white-space: nowrap;
}
[data-baseweb="tab"]:hover {
    color: var(--text-1) !important;
    background: rgba(255,255,255,0.05) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--primary) !important;
    color: #fff !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.35) !important;
}
/* Hide tab underline indicator */
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   BUTTONS
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
    padding: 0.45rem 1.1rem !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: var(--text-3) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
/* Primary button */
.stButton > button[kind="primary"],
.stButton > button.primary {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #fff !important;
    box-shadow: 0 0 12px var(--primary-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2563EB !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.4) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INPUTS & SELECTS
   ═══════════════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
    transition: var(--transition) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
    outline: none !important;
}
.stSelectbox > div > div > div:hover { border-color: var(--primary) !important; }
.stSlider > div > div > div { background: var(--primary) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════════════════════════════════ */
/* User message (right-aligned blue bubble) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child {
    background: var(--primary-dark) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.75rem 1.1rem !important;
    max-width: 72% !important;
    margin-left: auto !important;
    box-shadow: 0 2px 12px rgba(29,78,216,0.25) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: #fff !important;
}

/* AI message (left-aligned dark card with blue border) */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--primary) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    animation: fadeUp 0.3s ease forwards !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* Chat input bar */
[data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-full) !important;
    transition: var(--transition) !important;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.2) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow), 0 -4px 20px rgba(0,0,0,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
}
/* Send button inside chat input */
[data-testid="stChatInputSubmitButton"] button {
    background: var(--primary) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 0 8px var(--primary-glow) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   METRIC CARDS (GLASSMORPHISM)
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-metric-card {
    background: rgba(17,24,39,0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
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
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}
.dm-metric-value {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    margin-bottom: 0.25rem;
}
.dm-metric-label {
    font-size: 0.75rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.dm-metric-trend {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: var(--radius-full);
    display: inline-block;
    margin-top: 0.5rem;
}
.dm-metric-trend.up   { background: rgba(16,185,129,0.12); color: var(--success); }
.dm-metric-trend.down { background: rgba(244,63,94,0.12);  color: var(--danger);  }
.dm-metric-trend.stable{ background: rgba(139,92,246,0.12); color: var(--violet); }

/* ═══════════════════════════════════════════════════════════════════════════
   SOURCE CARDS (HORIZONTAL SCROLL STRIP)
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-source-strip {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 0.75rem 0;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
}
.dm-source-card {
    flex-shrink: 0;
    width: 240px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 0.85rem 1rem;
    transition: var(--transition);
    cursor: default;
}
.dm-source-card:hover {
    border-color: var(--primary);
    box-shadow: 0 0 12px var(--primary-glow);
    transform: translateY(-2px);
}
.dm-source-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dm-source-card-meta {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-3);
    margin-bottom: 0.5rem;
}
.dm-source-card-score {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--success);
    font-weight: 600;
}
.dm-source-card-excerpt {
    font-size: 0.72rem;
    color: var(--text-2);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CITATION PILLS
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-citation {
    display: inline-flex;
    align-items: center;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 5px;
    padding: 1px 7px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: #93C5FD;
    margin: 0 2px;
    cursor: default;
    transition: var(--transition);
}
.dm-citation:hover {
    background: rgba(59,130,246,0.2);
    border-color: var(--primary);
}

/* ═══════════════════════════════════════════════════════════════════════════
   LATENCY & STATUS BADGES
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: var(--radius-full);
    font-size: 0.7rem;
    font-family: var(--font-mono);
    font-weight: 500;
    border: 1px solid;
}
.dm-badge-blue   { background: rgba(59,130,246,0.1);  border-color: rgba(59,130,246,0.3);  color: #93C5FD; }
.dm-badge-green  { background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.3);  color: #6EE7B7; }
.dm-badge-amber  { background: rgba(245,158,11,0.1);  border-color: rgba(245,158,11,0.3);  color: #FCD34D; }
.dm-badge-violet { background: rgba(139,92,246,0.1);  border-color: rgba(139,92,246,0.3);  color: #C4B5FD; }

/* ═══════════════════════════════════════════════════════════════════════════
   INDEX STATUS BAR
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-status-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0;
    margin-bottom: 0.75rem;
    font-size: 0.78rem;
}
.dm-status-dot {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--success);
    font-family: var(--font-mono);
}
.dm-status-dot::before {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
    animation: pulse 2s infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INFO / ALERT BOXES
   ═══════════════════════════════════════════════════════════════════════════ */
.stAlert {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stInfo"] {
    background: rgba(59,130,246,0.06) !important;
    border-color: rgba(59,130,246,0.2) !important;
}
[data-testid="stSuccess"] {
    background: rgba(16,185,129,0.06) !important;
    border-color: rgba(16,185,129,0.2) !important;
}
[data-testid="stError"] {
    background: rgba(244,63,94,0.06) !important;
    border-color: rgba(244,63,94,0.2) !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,0.06) !important;
    border-color: rgba(245,158,11,0.2) !important;
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
}
[data-testid="stExpander"] summary:hover { color: var(--text-1) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   DATAFRAME / TABLE
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}
.stDataFrame thead tr th {
    background: var(--bg-elevated) !important;
    color: var(--text-2) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr td { color: var(--text-1) !important; font-size: 0.82rem !important; }
.stDataFrame tbody tr:hover td { background: rgba(255,255,255,0.03) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   SPINNER / PROGRESS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] > div {
    border-top-color: var(--primary) !important;
}
.stProgress > div > div { background: var(--primary) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   METRICS (st.metric)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    transition: var(--transition) !important;
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
[data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   DIVIDERS
   ═══════════════════════════════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION HEADERS
   ═══════════════════════════════════════════════════════════════════════════ */
.dm-section-header {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dm-section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    [data-baseweb="tab"] { padding: 6px 12px !important; font-size: 0.78rem !important; }
    .dm-source-card { width: 200px; }
}
</style>
"""
