# Contributing to Sreejita Framework

Thank you for your interest in contributing! We welcome all contributions, including bug reports, feature suggestions, and code improvements.

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/sreejita-framework.git
   cd sreejita-framework
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e '.[dev]'
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all functions
- Max line length: 100 characters
- Use meaningful variable names

### Testing
- Write tests for all new features
- Run tests before submitting PR:
  ```bash
  pytest tests/ --cov=sreejita
  ```
- Aim for >85% code coverage

### Commit Messages
- Use clear, descriptive commit messages
- Format: `[Category] Brief description`
- Example: `[Feature] Add new profiling module`

### Categories
- `[Feature]` - New functionality
- `[Fix]` - Bug fix
- `[Docs]` - Documentation
- `[Refactor]` - Code improvement
- `[Test]` - Test additions/improvements
- `[CI/CD]` - Infrastructure changes

## Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new code
   - Update CHANGELOG.md

2. **Submit PR**
   - Provide clear description of changes
   - Link related issues
   - Include screenshots/examples if relevant

3. **Code Review**
   - Address all review comments
   - Maintain respectful communication
   - Wait for approval before merging

## Reporting Issues

### Bug Reports
Include:
- Python version
- Error message and traceback
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Describe:
- Use case
- Proposed solution
- Alternative approaches

## License

By contributing, you agree your code will be licensed under the MIT License.

## Questions?

Feel free to open a Discussion or reach out to the maintainers.

Happy contributing! 🚀
