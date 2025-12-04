import pycountry

def get_iso3(name):
    # to fix problem of mismatch on map
    manual_map = {
        "Bolivia (Plurinational State of)": "BOL",
        "Iran (Islamic Republic of)": "IRN",
        "Iran, Islamic Rep.": "IRN",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Venezuela, RB": "VEN",
        "Viet Nam": "VNM",
        "Syrian Arab Republic": "SYR",
        "Lao People's Democratic Republic": "LAO",
        "Korea, Republic of": "KOR",
        "Korea, Rep.": "KOR",
        "Korea, Democratic People's Republic of": "PRK",
        "Korea, Dem. People's Rep.": "PRK",
        "Tanzania, United Republic of": "TZA",
        "Tanzania": "TZA",
        "Congo, Democratic Republic of the": "COD",
        "Congo, Dem. Rep.": "COD",
        "Congo, Rep.": "COG",
        "Moldova, Republic of": "MDA",
        "Micronesia (Federated States of)": "FSM",
        "United Kingdom of Great Britain and Northern Ireland": "GBR",
        "The former Yugoslav republic of Macedonia": "MKD",
        "Russia": "RUS", 
        "Russian Federation": "RUS",
        "Swaziland": "SWZ", 
        "Cape Verde": "CPV",
        "Bahamas, The": "BHS",
        "Gambia, The": "GMB",
        "Egypt, Arab Rep.": "EGY",
        "Yemen, Rep.": "YEM",
        "Kyrgyz Republic": "KGZ",
        "Slovak Republic": "SVK",
        "St. Lucia": "LCA",
        "St. Vincent and the Grenadines": "VCT",
        "Saint Vincent and the Grenadines": "VCT",
        "Cote d'Ivoire": "CIV",
        "Côte d'Ivoire": "CIV",
        "Niger": "NER",
        "Lao PDR": "LAO",
        "Micronesia, Fed. Sts.": "FSM",
        "Turkiye": "TUR",
        "Cabo Verde": "CPV",
        "Eswatini": "SWZ",
        "Czechia": "CZE"
    }
    
    if name in manual_map:
        return manual_map[name]
        
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except:
        return None
