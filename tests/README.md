# tests/

Test files for the flight app. These are development/debugging tests, not automated CI tests.

## Files

| File | What it tests |
|------|---------------|
| `test_db_save.py` | Database save operations |
| `test_json_extraction.py` | JSON parsing from LLM responses |
| `test_survey_save.py` | Survey response saving |
| `test_bridge.py` | Bridge module integration |
| `test_demo.py` | Demo mode functionality |
| `test_spotlight.py` | Spotlight feature |
| `test_listen.py` | LISTEN algorithm (legacy) |
| `test_lilo_fix.py` | LILO fixes (legacy) |
| `test_tutorial.py` | Tutorial overlay |

## Note

Some tests reference removed modules (LISTEN, LILO, components). These may need updating to match the current codebase.
