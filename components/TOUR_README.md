# Guided Tour Feature

A smooth, professional onboarding walkthrough for the Flight Ranker app.

## Features

✅ **Auto-starts on first visit** - New users see the tour automatically
✅ **Skippable** - Users can skip at any time
✅ **One-time only** - Won't show again after completion
✅ **Manual trigger** - Button in sidebar to restart anytime
✅ **Step-by-step navigation** - Back/Next buttons with progress indicator
✅ **Beautiful UI** - Gradient header, clean design
✅ **Session-based** - Uses Streamlit session state for reliability

## Implementation

### Files Created

1. **`components/simple_tour.py`** - Main tour component
   - `show_tour()` - Displays current tour step with navigation
   - `check_auto_start()` - Auto-starts tour for first-time users
   - `add_tour_button_to_sidebar()` - Adds manual trigger button
   - `get_tour_steps()` - Defines all 8 tour steps

2. **`components/guided_tour.py`** - Advanced Driver.js version (optional)
   - Uses Driver.js library for element highlighting
   - More complex but can highlight specific page elements
   - Not currently integrated (simple version preferred for Streamlit)

3. **`components/tour_integration_example.py`** - Integration examples

### Integration in app.py

Added 4 lines after token validation:

```python
# Import
from components.simple_tour import show_tour, check_auto_start, add_tour_button_to_sidebar

# Initialize and show (lines 791-793)
check_auto_start(skip_for_demo=True)  # Auto-start for first-time users
show_tour()  # Display current tour step if active
add_tour_button_to_sidebar()  # Add manual trigger button in sidebar
```

## Tour Steps

The tour has **8 steps** covering:

1. **Welcome** - Overview of what the app does
2. **Describe Your Flight** - How to enter flight preferences
3. **Search Buttons** - Difference between regular and AI search
4. **Review Results** - Understanding the results table
5. **Rank Your Selections** - How to select and rank flights
6. **LILO Mode** - Optional advanced preference learning
7. **Validation & Survey** - Final steps in the process
8. **You're Ready!** - Completion with quick tips

Each step includes:
- Clear title
- Detailed description with examples
- Step progress indicator (e.g., "Step 2 of 8")
- Navigation buttons (Back/Next/Skip)

## User Experience

### First Visit
1. User opens app with valid token
2. Tour automatically starts on Step 1
3. User can click "Next" to proceed or "Skip" to close
4. After completion/skip, tour won't show again

### Returning Users
- Tour doesn't auto-start
- Can manually trigger via "🎓 Show Tutorial" button in sidebar
- Starts from Step 1 and goes through all 8 steps

### DEMO Token
- Tour is automatically skipped for DEMO token users
- Assumes they're exploring freely and don't need onboarding
- Can still trigger manually via sidebar button

## Configuration

### Skip Auto-Start for DEMO Token

```python
check_auto_start(skip_for_demo=True)  # Current setting
```

To show tour for DEMO users too:

```python
check_auto_start(skip_for_demo=False)
```

### Modify Tour Steps

Edit `get_tour_steps()` in `components/simple_tour.py`:

```python
def get_tour_steps():
    return [
        (
            "Step Title",
            "Step description with **markdown** support"
        ),
        # Add more steps...
    ]
```

### Custom Styling

The tour uses inline CSS for styling. Modify in `show_tour()` function:

```python
st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 30px; border-radius: 12px 12px 0 0; color: white;">
    <!-- Header styling -->
</div>
""", unsafe_allow_html=True)
```

## Session State Variables

The tour uses these session state variables:

- `tour_active` (bool) - Whether tour is currently running
- `tour_step` (int) - Current step index (0-7)
- `tour_completed` (bool) - Whether user has completed/skipped tour

## Advanced: Driver.js Version

For element-highlighting functionality, use the Driver.js version:

1. Use `components/guided_tour.py` instead of `simple_tour.py`
2. Define CSS selectors for elements to highlight
3. More complex but provides better visual guidance

**Note:** Currently not integrated due to Streamlit's dynamic element rendering making selector targeting unreliable.

## Troubleshooting

### Tour doesn't auto-start
- Check `tour_completed` in session state isn't True
- Verify `check_auto_start()` is called before `show_tour()`
- Clear session state or use incognito window to test

### Tour shows every time
- Check that `tour_completed` is being set to True
- Verify session state persists during app usage

### Styling issues
- Ensure `unsafe_allow_html=True` is set for markdown with HTML
- Check browser console for CSS errors

## Future Enhancements

Possible improvements:

- [ ] Persist tour completion to database (not just session)
- [ ] Track which steps users skip most
- [ ] Add interactive elements (e.g., "try searching now")
- [ ] Localization for multiple languages
- [ ] Tour customization based on user role
- [ ] Analytics on tour completion rate
