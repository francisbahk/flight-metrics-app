"""
CSS and JS string constants used throughout the app.
Inject with st.markdown(CSS, unsafe_allow_html=True).
"""

GLOBAL_CSS = """
<style>
    /* Prevent auto-scroll on interactions */
    html {
        scroll-behavior: auto !important;
        overflow-anchor: none !important;
    }
    body {
        overflow-anchor: none !important;
    }
    .main {
        overflow-anchor: none !important;
    }

    /* Prevent checkboxes from triggering scroll */
    .stCheckbox {
        overflow-anchor: none !important;
    }
    .stCheckbox input[type="checkbox"] {
        overflow-anchor: none !important;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .flight-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .shortlist-area {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 1.5rem;
        min-height: 400px;
    }
    /* Compact flight list */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="column"] {
        padding: 0.2rem !important;
    }
    .stMarkdown p {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }
    /* Make checkboxes 15% larger */
    input[type="checkbox"] {
        transform: scale(1.15);
        cursor: pointer;
    }
</style>
<script>
    // Prevent scrolling when checkboxes are clicked
    (function() {
        let scrollPosition = 0;

        function attachScrollPrevention() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(function(checkbox) {
                if (!checkbox.hasAttribute('data-scroll-prevention')) {
                    checkbox.setAttribute('data-scroll-prevention', 'true');

                    checkbox.addEventListener('mousedown', function() {
                        scrollPosition = window.scrollY;
                    });

                    checkbox.addEventListener('change', function() {
                        setTimeout(function() {
                            window.scrollTo(0, scrollPosition);
                        }, 0);
                    });

                    checkbox.addEventListener('click', function() {
                        setTimeout(function() {
                            window.scrollTo(0, scrollPosition);
                        }, 10);
                    });
                }
            });
        }

        attachScrollPrevention();

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', attachScrollPrevention);
        }

        const observer = new MutationObserver(attachScrollPrevention);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
</script>
"""

TEXTAREA_CSS = """
<style>
    .stTextArea {
        position: relative !important;
    }
    .stTextArea textarea {
        background-color: transparent !important;
        position: relative !important;
        z-index: 2 !important;
    }
    .anim-placeholder {
        position: absolute !important;
        top: 12px !important;
        left: 12px !important;
        right: 12px !important;
        bottom: 12px !important;
        color: #94a3b8 !important;
        font-family: 'Source Code Pro', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        pointer-events: none !important;
        white-space: pre-wrap !important;
        overflow: hidden !important;
        z-index: 1 !important;
    }
    .anim-placeholder.hide {
        display: none !important;
    }
</style>
"""

PROMPT_SPACING_CSS = """
<style>
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.25rem !important;
    }
</style>
"""

NEON_METRIC_CSS = """
<style>
    @keyframes redNeonPulse {
        0% {
            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
            border-color: #ff4444;
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 6px #ff4444, 0 0 12px #ff4444, 0 0 18px #ff4444;
            border-color: #ff6666;
            transform: scale(1.05);
        }
        100% {
            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
            border-color: #ff4444;
            transform: scale(1);
        }
    }
    .neon-metric-box {
        display: inline-block;
        animation: redNeonPulse 2s ease-in-out infinite;
        padding: 3px 8px;
        border-radius: 5px;
        margin: 0 3px;
        border: 2px solid #ff4444;
        box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
        transition: all 0.3s ease;
    }
</style>
"""
