# AI Coding Assistants Instructions for desispec

These instructions are for GitHub Copilot, Claude Code, and similar coding assistants.

## Project Overview
desispec is a Python package for Dark Energy Spectroscopic Instrument (DESI) data processing. It handles spectral data extraction, calibration, and Slurm batch job workflow.

## Code Style & Standards

### Python Style
- Follow PEP 8 guidelines
- Use 4-space indentation
- Do not include type hints
- Only use ASCII characters in code, comments, and docstrings; do not use unicode symbols, emojis, or non-latin letters
- Keep lines under 120 characters when possible
- Prefer functions that return values rather than modifying arguments in place
- Classes may be used for data structures, but avoid complex class hierarchies and classes that manage state or contain algorithmic logic
- Prioritize clarity over cleverness; write code that is easy to read and understand by non-expert collaborators
- Support Python 3.10 and newer

### Documentation
- Provide clear docstrings with brief descriptions, input parameters, and return values
- Use NumPy or Google style for docstrings
- Add comments for complex logic and non-obvious implementations

## Key Guidelines

### When Writing New Code
- **Imports**: Keep imports organized - standard library first, then third-party, then local imports
- **Error Handling**: Use appropriate exceptions and provide meaningful error messages
- **Logging**: Use the desiutil.log module for debug/info messages rather than print statements
- **Testing**: Include unit tests for new functions in the appropriate test directory

### Policy When Modifying Existing Code
- Check existing code patterns before suggesting alternatives
- Make ONLY the changes explicitly requested
- Update related documentation when making changes
- Do NOT refactor, improve, or fix unrelated code without asking
- Do NOT reorder pre-existing imports, dictionaries, or other elements unless asked
- If you see something that should be changed, ASK first
- Do NOT break backwards compatibility without explicit instructions to do so
- Do NOT add new 3rd party dependencies without explicit instructions to do so

### Project Structure
- `/py/` - Main Python package code
- `/py/desispec/tests/` - Unit tests
- `/bin/` - Command line scripts
- `/doc/` - Documentation
- `/etc/` - Specialized files; ignore these

desispec supports installation with `pip install .`, and also supports direct usage by adding the `/py/` directory to $PYTHONPATH and the `/bin/` directory to $PATH. When modifying code, ensure that it remains compatible with both usage patterns.

### Common Modules to Reference
- `desispec.io` - File I/O operations
- `desispec.scripts` - Functions to be wrapped by the command-line scripts in `/bin/`
- `desispec.workflow` - Batch job workflow management
- `desispec.spectra` - Spectral data structures and manipulation

## Testing & Development
- Use pytest for unit tests
- Place tests in `/py/desispec/tests/` with naming pattern `test_*.py`

## Git Conventions
- Use clear, descriptive commit messages
- Reference issues and pull requests appropriately
- Test locally before committing
- Keep commits focused on single logical changes

