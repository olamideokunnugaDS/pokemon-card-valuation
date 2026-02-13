# Contributing to Pokémon Card Valuation Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to this research project.

## Code of Conduct

This project follows academic research ethics and professional coding standards. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the issue tracker
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment details

### Submitting Changes

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes following our coding standards
4. Add tests for new functionality
5. Run the test suite (`make test`)
6. Commit with clear messages (`git commit -m "Add feature X"`)
7. Push to your fork (`git push origin feature/your-feature-name`)
8. Create a Pull Request

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Format code with `black` and `isort`
- Document all functions with docstrings

Example:
```python
def calculate_embedding(
    image: torch.Tensor,
    metadata: Dict[str, Any],
    normalize: bool = True
) -> torch.Tensor:
    """
    Calculate condition embedding from card image.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        metadata: Card metadata dictionary
        normalize: Whether to normalize the embedding
        
    Returns:
        Condition embedding tensor of shape (embedding_dim,)
        
    Raises:
        ValueError: If image dimensions are invalid
    """
    pass
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and PRs when relevant

Example:
```
Add CNN interpretability visualizations

- Implement Grad-CAM for feature attribution
- Add saliency map generation
- Create visualization utilities

Closes #42
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/unit/test_vision_module.py
```

### Writing Tests

- Write unit tests for all new functions
- Ensure test coverage > 80%
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

Example:
```python
def test_vision_encoder_output_shape():
    """Test that vision encoder produces correct output shape."""
    # Arrange
    encoder = VisionEncoder(embedding_dim=256)
    input_image = torch.randn(1, 3, 224, 224)
    
    # Act
    output = encoder(input_image)
    
    # Assert
    assert output.shape == (1, 256), f"Expected shape (1, 256), got {output.shape}"
```

## Documentation

### Code Documentation

- All modules, classes, and functions must have docstrings
- Use Google-style docstrings
- Include examples for complex functions
- Update README when adding new features

### Research Documentation

When contributing to research methodology:

- Document theoretical justifications
- Cite relevant papers
- Explain design decisions
- Include ablation study results

## Review Process

1. Code review by at least one maintainer
2. All tests must pass
3. Code style checks must pass
4. Documentation must be updated
5. No merge conflicts with main branch

## Questions?

Feel free to open an issue for questions or clarifications. For research collaboration inquiries, contact the project maintainer directly.

Thank you for contributing!
