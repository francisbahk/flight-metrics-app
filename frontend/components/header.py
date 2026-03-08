"""
Header and token gate components.
"""
import streamlit as st
import streamlit.components.v1 as components


HEADER_HTML = """<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .main-title {
        text-align: center;
        width: 100%;
        padding: 20px 0;
        font-size: 2.5rem;
        font-weight: 600;
        transition: all 0.8s ease-in-out;
    }
    .title-wrapper {
        display: inline-block;
    }
    #changing-word {
        display: inline-block;
        transition: all 2s ease-in-out;
        font-style: italic;
    }
    #ai-prefix {
        display: inline-block;
        transition: opacity 1.5s ease-in-out, transform 1.5s ease-in-out;
        font-style: italic;
    }
    #flight-word {
        display: inline-block;
    }
    .subtitle {
        text-align: center;
        width: 100%;
        margin-top: 10px;
        font-size: 1.1rem;
        color: #555;
    }
    .subtitle span {
        display: inline-block;
        transition: all 0.8s ease-in-out;
        font-style: italic;
    }
</style>
</head>
<body>

<div class="main-title">
    <div class="title-wrapper">
        <span>✈️ <span id="ai-prefix" style="opacity: 0; transform: translateX(-20px);"></span> <span id="flight-word">Flight</span> <span id="changing-word">Ranker</span></span>
    </div>
</div>

<div class="subtitle" id="subtitle-content">
    Share your flight preferences to <span id="word1">help</span> <span id="word2">build</span> <span id="word3">better</span> personalized <span id="word4">ranking</span> <span id="word5">systems</span>
</div>

<script>
(function() {
    const CYCLE_DURATION = 30000;
    const START_DELAY = 1000;
    const startTime = Date.now() + START_DELAY;

    var titleAnimating = false;
    var subtitleAnimating = false;
    var lastPhase = 0;

    function smoothFlip(element, newText, duration) {
        element.style.transform = 'rotateX(90deg)';
        element.style.opacity = '0';
        setTimeout(function() {
            element.textContent = newText;
            element.style.transform = 'rotateX(0deg)';
            element.style.opacity = '1';
        }, duration / 2);
    }

    function animateTitle() {
        const changingWord = document.getElementById('changing-word');
        const aiPrefix = document.getElementById('ai-prefix');
        if (!changingWord) return;

        const now = Date.now();
        if (now < startTime) return;

        const cyclePosition = ((now - startTime) % CYCLE_DURATION) / CYCLE_DURATION;
        const currentPhase = Math.floor(cyclePosition * 100);

        if (cyclePosition < 0.25) {
            if (lastPhase >= 25 && changingWord.textContent !== 'Ranker') {
                titleAnimating = true;
                aiPrefix.style.opacity = '0';
                aiPrefix.style.transform = 'translateX(-20px)';
                changingWord.style.transform = 'rotateX(90deg)';
                changingWord.style.opacity = '0';
                setTimeout(function() {
                    changingWord.textContent = 'Ranker';
                    changingWord.style.transform = 'rotateX(0deg)';
                    changingWord.style.opacity = '1';
                    aiPrefix.textContent = '';
                    titleAnimating = false;
                }, 600);
            }
        } else if (cyclePosition >= 0.25 && cyclePosition < 0.32) {
            if (!titleAnimating && changingWord.textContent === 'Ranker') {
                titleAnimating = true;
                changingWord.style.transform = 'translateX(-30px)';
                changingWord.style.opacity = '0';
                setTimeout(function() {
                    changingWord.textContent = 'Recommendations';
                    changingWord.style.transform = 'translateX(30px)';
                    setTimeout(function() {
                        changingWord.style.transform = 'translateX(0)';
                        changingWord.style.opacity = '1';
                    }, 50);
                }, 600);
                setTimeout(function() {
                    aiPrefix.textContent = 'AI-Driven';
                    aiPrefix.style.opacity = '1';
                    aiPrefix.style.transform = 'translateX(0)';
                    titleAnimating = false;
                }, 1800);
            }
        } else if (cyclePosition >= 0.80 && cyclePosition < 0.85) {
            if (!titleAnimating && aiPrefix.style.opacity !== '0') {
                titleAnimating = true;
                aiPrefix.style.opacity = '0';
                aiPrefix.style.transform = 'translateX(-20px)';
                setTimeout(function() {
                    changingWord.style.transform = 'translateX(30px)';
                    changingWord.style.opacity = '0';
                    setTimeout(function() {
                        changingWord.textContent = 'Ranker';
                        changingWord.style.transform = 'translateX(30px)';
                        setTimeout(function() {
                            changingWord.style.transform = 'translateX(0)';
                            changingWord.style.opacity = '1';
                            aiPrefix.textContent = '';
                            titleAnimating = false;
                        }, 50);
                    }, 600);
                }, 600);
            }
        }
        lastPhase = currentPhase;
    }

    function animateSubtitleWord(wordId, newText, delay) {
        setTimeout(function() {
            const word = document.getElementById(wordId);
            if (word) {
                word.style.transform = 'rotateX(90deg)';
                word.style.opacity = '0';
                setTimeout(function() {
                    word.textContent = newText;
                    word.style.transform = 'rotateX(0deg)';
                    word.style.opacity = '1';
                }, 600);
            }
        }, delay);
    }

    function animateSubtitle() {
        if (titleAnimating) return;
        const now = Date.now();
        if (now < startTime) return;
        const cyclePosition = ((now - startTime) % CYCLE_DURATION) / CYCLE_DURATION;

        if (cyclePosition >= 0.40 && cyclePosition < 0.45) {
            if (!subtitleAnimating) {
                subtitleAnimating = true;
                animateSubtitleWord('word1', 'receive', 0);
                animateSubtitleWord('word2', 'smart', 800);
                animateSubtitleWord('word3', 'fast', 1600);
                animateSubtitleWord('word4', 'flight', 2400);
                animateSubtitleWord('word5', 'results', 3200);
                setTimeout(function() { subtitleAnimating = false; }, 4400);
            }
        } else if (cyclePosition >= 0.88 && cyclePosition < 0.93) {
            if (!subtitleAnimating && !titleAnimating) {
                subtitleAnimating = true;
                animateSubtitleWord('word1', 'help', 0);
                animateSubtitleWord('word2', 'build', 800);
                animateSubtitleWord('word3', 'better', 1600);
                animateSubtitleWord('word4', 'ranking', 2400);
                animateSubtitleWord('word5', 'systems', 3200);
                setTimeout(function() { subtitleAnimating = false; }, 4400);
            }
        }
    }

    setInterval(function() {
        animateTitle();
        animateSubtitle();
    }, 100);
})();
</script>

</body>
</html>"""


def render_header():
    """Render the animated title/subtitle header iframe."""
    components.html(HEADER_HTML, height=150)
    st.info("**Note:** This website is part of a pilot data-collection study. The information collected will be used to improve flight search tools.")


def render_token_gate():
    """
    Validate and display token status. Calls st.stop() if access is denied.
    Must be called after session state token fields are populated.

    Phase tokens (PHASEONE_<prolific_id>, PHASETWO_<prolific_id>) pass through
    silently — the Prolific ID gate already handled identity verification.
    """
    from phases import is_phase_token, is_phase_url

    token = st.session_state.token or ''

    # Phase participants: verified via Prolific gate — pass through silently
    if is_phase_token(token):
        return

    # Fallback: if the URL is a phase URL, show the gate instead of "Access Denied"
    try:
        _params = st.experimental_get_query_params()
        _url_id = (_params.get('id', [''])[0] or '').upper()
    except Exception:
        _url_id = ''

    if is_phase_url(_url_id):
        if not st.session_state.get('prolific_id'):
            from frontend.pages.prolific_gate import render_prolific_id_gate
            render_prolific_id_gate(_url_id)
            st.stop()
        return

    # Special admin/demo tokens
    if token.upper() in ["DEMO", "DATA"]:
        st.session_state.token_valid = True
        if token.upper() == "DEMO":
            st.info("🎯 **Demo Mode** - Explore the flight search tool freely!")
        else:
            st.info("📊 **Data Collection Mode** - Unlimited submissions enabled!")
        return

    # No token at all
    if not token:
        st.error("❌ **Access Denied: No Token Provided**")
        st.warning("This study requires a unique access token. Please use the link provided to you by the researchers, or use `?id=DEMO` for demo mode.")
        st.stop()

    # Legacy token — check validity
    if not st.session_state.token_valid:
        if 'already used' in st.session_state.token_message.lower():
            st.error("❌ **Access Not Granted: This token has already been used**")
            st.warning("This access link can only be used once. If you need to participate again, please contact the research team for a new token.")
        else:
            st.error(f"❌ **Access Denied: {st.session_state.token_message}**")
            st.warning("Please check your access link and try again, or contact the researchers if you believe this is an error.")
        st.stop()
