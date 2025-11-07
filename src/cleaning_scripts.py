def standardize_body(body):
    body = str(body).lower()
    
    # Check for keywords in order of specificity
    if 'convertible' in body:
        return 'convertible'
    elif 'van' in body or 'minivan' in body:
        return 'van'
    elif any(x in body for x in ['crew', 'double', 'quad', 'mega', 'king', 'super', 'extended', 'regular', 'access', 'club', 'cab plus', 'xtracab', 'crewmax']):
        return 'truck'
    elif 'coupe' in body or 'koup' in body:
        return 'coupe'
    elif 'sedan' in body or 'g sedan' in body or 'g37' in body:
        return 'sedan'
    elif 'wagon' in body:
        return 'wagon'
    elif 'hatchback' in body:
        return 'hatchback'
    elif 'suv' in body:
        return 'suv'
    else:
        return 'other'