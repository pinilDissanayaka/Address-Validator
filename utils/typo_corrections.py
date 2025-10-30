"""
Common typo corrections for Philippine place names.
This module provides dictionaries for known typos and variations in spelling.
"""

# Common barangay name typos and variations
BARANGAY_TYPOS = {
    'licenaa': 'licena',
    'licensaa': 'licena',
    'queebiawan': 'quebiauan',
    'quebiawan': 'quebiauan',
    'hippodromo': 'hippodromo',
    'hipodromo': 'hippodromo',
    'cordoba': 'cordova',
    'palao': 'pala-o',
    'pala o': 'pala-o',
    'pulungbulu': 'pulung bulu',
    'pulung-bulu': 'pulung bulu',
}

# Common city name typos and variations
CITY_TYPOS = {
    'calocan': 'caloocan',
    'caloocan': 'caloocan',
    'mandaluyong': 'mandaluyong',
    'muntinlupa': 'muntinlupa',
    'muntilupa': 'muntinlupa',
    'paraaque': 'parañaque',
    'paranaque': 'parañaque',
    'cordoba': 'cordova',
    'san fernando': 'city of san fernando',
    'cagayan de oro': 'city of cagayan de oro',
    'lapu lapu': 'city of lapu-lapu',
    'lapu-lapu': 'city of lapu-lapu',
}

# Common province name typos and variations
PROVINCE_TYPOS = {
    'la union': 'la union',
    'launion': 'la union',
    'compostela valley': 'davao de oro',
    'davao del sur': 'davao del sur',
    'davaodelsur': 'davao del sur',
    'misamis oriental': 'misamis oriental',
    'misamis occidental': 'misamis occidental',
    'agusan del norte': 'agusan del norte',
    'agusan del sur': 'agusan del sur',
}


def correct_typo(text: str, component_type: str = 'barangay') -> str:
    """
    Correct known typos in Philippine place names.
    
    Args:
        text: The text to correct
        component_type: Type of component ('barangay', 'city', 'province')
        
    Returns:
        Corrected text or original if no correction found
    """
    if not text:
        return text
    
    text_lower = text.lower().strip()
    
    if component_type == 'barangay' and text_lower in BARANGAY_TYPOS:
        return BARANGAY_TYPOS[text_lower]
    elif component_type == 'city' and text_lower in CITY_TYPOS:
        return CITY_TYPOS[text_lower]
    elif component_type == 'province' and text_lower in PROVINCE_TYPOS:
        return PROVINCE_TYPOS[text_lower]
    
    return text


def apply_common_corrections(parsed_address) -> object:
    """
    Apply common typo corrections to a parsed address object.
    
    Args:
        parsed_address: ParsedAddress object or similar with province, city, barangay attributes
        
    Returns:
        Modified parsed_address object
    """
    if hasattr(parsed_address, 'province') and parsed_address.province:
        parsed_address.province = correct_typo(parsed_address.province, 'province')
    
    if hasattr(parsed_address, 'city') and parsed_address.city:
        parsed_address.city = correct_typo(parsed_address.city, 'city')
    
    if hasattr(parsed_address, 'barangay') and parsed_address.barangay:
        parsed_address.barangay = correct_typo(parsed_address.barangay, 'barangay')
    
    return parsed_address
