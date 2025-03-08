"""TEM image analyzers module."""

from tem_agent.analyzers.finfet import FinfetAnalyzer

# Dictionary mapping analyzer types to their classes
ANALYZER_TYPES = {
    'finfet': FinfetAnalyzer,
    'fin': FinfetAnalyzer
}

def create_analyzer(analyzer_type, **kwargs):
    """Create an analyzer of the specified type.
    
    Args:
        analyzer_type: Type of analyzer to create ('finfet', 'fin')
        **kwargs: Additional parameters to pass to the analyzer constructor
        
    Returns:
        An analyzer instance of the requested type
        
    Raises:
        ValueError: If the analyzer type is not supported
    """
    analyzer_type = analyzer_type.lower()
    
    if analyzer_type in ANALYZER_TYPES:
        return ANALYZER_TYPES[analyzer_type](**kwargs)
    else:
        supported_types = ', '.join(ANALYZER_TYPES.keys())
        raise ValueError(f"Unsupported analyzer type: {analyzer_type}. "
                        f"Supported types are: {supported_types}")
                        
__all__ = ['FinfetAnalyzer', 'create_analyzer', 'ANALYZER_TYPES'] 