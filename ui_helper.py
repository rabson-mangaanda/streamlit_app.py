import streamlit as st

# Custom CSS with Tailwind-like utilities
CUSTOM_STYLES = """
<style>
    /* Base styles */
    .stApp {
        max-width: 100%;
    }
    
    /* Tailwind-like utilities */
    .container { width: 100%; margin: auto; }
    .card { background-color: white; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
    .header { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #1f2937; }
    .subheader { font-size: 1.25rem; color: #4b5563; margin-bottom: 0.5rem; }
    .badge { padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 500; }
    .badge-blue { background-color: #3b82f6; color: white; }
    .badge-green { background-color: #10b981; color: white; }
    .badge-red { background-color: #ef4444; color: white; }
    .stat-card { background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 1rem; margin: 0.5rem 0; }
    .flex { display: flex; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
</style>
"""

def inject_custom_css():
    """Inject custom CSS into the app"""
    st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

def card(title, content, color="blue"):
    """Create a styled card component"""
    st.markdown(f"""
        <div class="card">
            <div class="header" style="color: var(--primary-color);">{title}</div>
            <div>{content}</div>
        </div>
    """, unsafe_allow_html=True)

def stat_card(label, value, delta=None):
    """Create a metric card with optional delta"""
    if delta:
        delta_color = "text-green-500" if float(delta) > 0 else "text-red-500"
        delta_html = f'<span class="{delta_color}">({delta}%)</span>'
    else:
        delta_html = ""
    
    st.markdown(f"""
        <div class="stat-card">
            <div class="subheader">{label}</div>
            <div class="flex">
                <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
                <div style="margin-left: 0.5rem;">{delta_html}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def badge(text, color="blue"):
    """Create a colored badge"""
    return f'<span class="badge badge-{color}">{text}</span>'

def section_header(title, description=None):
    """Create a styled section header"""
    st.markdown(f"""
        <div style="margin: 2rem 0 1rem 0;">
            <div class="header">{title}</div>
            {f'<div style="color: #6b7280;">{description}</div>' if description else ''}
        </div>
    """, unsafe_allow_html=True)
