import numpy as np
import pandas as pd


def preprocess_data(raw_data, name):
    if name == 'orders':
        num_indices = [0, 1, 3, 6]
        cat_indices = [2, 5]
        delimiter = '|'
    elif name == 'partsupp':
        num_indices = [0, 1, 2, 3]
        cat_indices = []
        delimiter = '|'
    else:
        raise ValueError("Unsupported dataset")

    combined_data = []

    orders_priority_dict = {
        '1-URGENT': 1,
        '2-HIGH': 2,
        '3-MEDIUM': 3,
        '4-NOT SPECIFIED': 4,
        '5-LOW': 5
    }

    orders_status_dict = {
        'F': 1,
        'O': 2,
        'P': 3
    }

    for i in range(len(raw_data)):
        pre = raw_data[i].strip('\n').split(delimiter)

        if 'orders' in name:
            pre[5] = orders_priority_dict.get(pre[5], pre[5])
            pre[2] = orders_status_dict.get(pre[2], pre[2])
            pre[6] = int(pre[6][10:])

        # Collecting numerical and categorical data separately
        num_list = [pre[j] for j in num_indices]
        cate_list = [pre[j] for j in cat_indices]

        combined_data.append(num_list + cate_list)

    # new_num_indices = np.arange(len(num_indices)).tolist()
    # new_cat_indices = np.arange(len(num_indices), len(num_indices) + len(cat_indices)).tolist()

    return combined_data


def preprocess_flight(raw_data):
    flight_num_indices = [10, 11, 12, 13]
    flight_cat_indices = [2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19]

    flight_date_dict = {
     '2012-11-01': 1,  '2012-11-02': 2,  '2012-11-03': 3,  '2012-11-04': 4,
     '2012-11-05': 5,  '2012-11-06': 6,  '2012-11-07': 7,  '2012-11-08': 8,
     '2012-11-09': 9,  '2012-11-10': 10, '2012-11-11': 11, '2012-11-12': 12,
     '2012-11-13': 13, '2012-11-14': 14, '2012-11-15': 15, '2012-11-16': 16,
     '2012-11-17': 17, '2012-11-18': 18, '2012-11-19': 19, '2012-11-20': 20,
     '2012-11-21': 21, '2012-11-22': 22, '2012-11-23': 23, '2012-11-24': 24,
     '2012-11-25': 25, '2012-11-26': 26, '2012-11-27': 27, '2012-11-28': 28,
     '2012-11-29': 29, '2012-11-30': 30, '2012-12-01': 31, '2012-12-02': 32,
     '2012-12-03': 33, '2012-12-04': 34, '2012-12-05': 35, '2012-12-06': 36,
     '2012-12-07': 37, '2012-12-08': 38, '2012-12-09': 39, '2012-12-10': 40,
     '2012-12-11': 41, '2012-12-12': 42, '2012-12-13': 43, '2012-12-14': 44,
     '2012-12-15': 45, '2012-12-16': 46, '2012-12-17': 47, '2012-12-18': 48,
     '2012-12-19': 49, '2012-12-20': 50, '2012-12-21': 51, '2012-12-22': 52,
     '2012-12-23': 53, '2012-12-24': 54, '2012-12-25': 55, '2012-12-26': 56,
     '2012-12-27': 57, '2012-12-28': 58, '2012-12-29': 59, '2012-12-30': 60,
     '2012-12-31': 61
    }

    carrier_dict = {
        'AA': 1,
        'AS': 2,
        'B6': 3,
        'DL': 4,
        'EV': 5,
        'F9': 6,
        'FL': 7,
        'HA': 8,
        'MQ': 9,
        'OO': 10,
        'UA': 11,
        'US': 12,
        'VX': 13,
        'WN': 14,
        'YV': 15
    }

    origin_state_dict = {
        'AK': 1,
        'AL': 2,
        'AR': 3,
        'AZ': 4,
        'CA': 5,
        'CO': 6,
        'CT': 7,
        'FL': 8,
        'GA': 9,
        'HI': 10,
        'IA': 11,
        'ID': 12,
        'IL': 13,
        'IN': 14,
        'KS': 15,
        'KY': 16,
        'LA': 17,
        'MA': 18,
        'MD': 19,
        'ME': 20,
        'MI': 21,
        'MN': 22,
        'MO': 23,
        'MS': 24,
        'MT': 25,
        'NC': 26,
        'ND': 27,
        'NE': 28,
        'NH': 29,
        'NJ': 30,
        'NM': 31,
        'NV': 32,
        'NY': 33,
        'OH': 34,
        'OK': 35,
        'OR': 36,
        'PA': 37,
        'PR': 38,
        'RI': 39,
        'SC': 40,
        'SD': 41,
        'TN': 42,
        'TT': 43,
        'TX': 44,
        'UT': 45,
        'VA': 46,
        'VI': 47,
        'VT': 48,
        'WA': 49,
        'WI': 50,
        'WV': 51,
        'WY': 52
    }

    origin_state_name_dict = {
        'Alabama': 1,
        'Alaska': 2,
        'Arizona': 3,
        'Arkansas': 4,
        'California': 5,
        'Colorado': 6,
        'Connecticut': 7,
        'Florida': 8,
        'Georgia': 9,
        'Hawaii': 10,
        'Idaho': 11,
        'Illinois': 12,
        'Indiana': 13,
        'Iowa': 14,
        'Kansas': 15,
        'Kentucky': 16,
        'Louisiana': 17,
        'Maine': 18,
        'Maryland': 19,
        'Massachusetts': 20,
        'Michigan': 21,
        'Minnesota': 22,
        'Mississippi': 23,
        'Missouri': 24,
        'Montana': 25,
        'Nebraska': 26,
        'Nevada': 27,
        'New Hampshire': 28,
        'New Jersey': 29,
        'New Mexico': 30,
        'New York': 31,
        'North Carolina': 32,
        'North Dakota': 33,
        'Ohio': 34,
        'Oklahoma': 35,
        'Oregon': 36,
        'Pennsylvania': 37,
        'Puerto Rico': 38,
        'Rhode Island': 39,
        'South Carolina': 40,
        'South Dakota': 41,
        'Tennessee': 42,
        'Texas': 43,
        'U.S. Pacific Trust Territories and Possessions': 44,
        'U.S. Virgin Islands': 45,
        'Utah': 46,
        'Vermont': 47,
        'Virginia': 48,
        'Washington': 49,
        'West Virginia': 50,
        'Wisconsin': 51,
        'Wyoming': 52
    }

    selected_numeric_columns = raw_data.iloc[:, flight_num_indices]
    selected_categorical_columns = raw_data.iloc[:, flight_cat_indices]

    combined_selection = pd.concat([selected_numeric_columns, selected_categorical_columns], axis=1)

    combined_selection['FlightDate'] = combined_selection['FlightDate'].map(flight_date_dict)
    combined_selection['UniqueCarrier'] = combined_selection['UniqueCarrier'].map(carrier_dict)
    combined_selection['Carrier'] = combined_selection['Carrier'].map(carrier_dict)
    combined_selection['OriginState'] = combined_selection['OriginState'].map(origin_state_dict)
    combined_selection['OriginStateName'] = combined_selection['OriginStateName'].map(origin_state_name_dict)

    return combined_selection


def preprocess_nypd(raw_data):
    nypd_num_indices = [0, 3, 4, 18, 21, 30, 31, 32, 33]
    nypd_cat_indices = [1, 2, 7, 10, 11, 12, 13, 15, 17, 20, 27, 28, 29]

    boro_nm_dict = {
        'BRONX': 1, 'BROOKLYN': 2, 'MANHATTAN': 3, 'QUEENS': 4, 'STATEN ISLAND': 5
    }

    crm_atpt_cptd_cd_dict = {
        'ATTEMPTED': 0, 'COMPLETED': 1
    }

    juris_desc_dict = {
        'AMTRACK': 1,
        'DEPT OF CORRECTIONS': 2,
        'HEALTH & HOSP CORP': 3,
        'LONG ISLAND RAILRD': 4,
        'METRO NORTH': 5,
        'N.Y. HOUSING POLICE': 6,
        'N.Y. POLICE DEPT': 7,
        'N.Y. STATE PARKS': 8,
        'N.Y. STATE POLICE': 9,
        'N.Y. TRANSIT POLICE': 10,
        'NEW YORK CITY SHERIFF OFFICE': 11,
        'NYS DEPT TAX AND FINANCE': 12,
        'NYC PARKS': 13,
        'OTHER': 14,
        'PORT AUTHORITY': 15,
        'STATN IS RAPID TRANS': 16,
        'TRI-BORO BRDG TUNNL': 17,
        'U.S. PARK POLICE': 18
    }

    law_cat_cd_dict = {
        'FELONY': 1, 'MISDEMEANOR': 2, 'VIOLATION': 3
    }

    ofns_desc_dict = {
        'ABORTION': 1,
        'ADMINISTRATIVE CODE': 2,
        'AGRICULTURE & MRKTS LAW-UNCLASSIFIED': 3,
        'ALCOHOLIC BEVERAGE CONTROL LAW': 4,
        'ANTICIPATORY OFFENSES': 5,
        'ARSON': 6,
        'ASSAULT 3 & RELATED OFFENSES': 7,
        "BURGLAR'S TOOLS": 8,
        'BURGLARY': 9,
        'CHILD ABANDONMENT/NON SUPPORT': 10,
        'CRIMINAL MISCHIEF & RELATED OF': 11,
        'CRIMINAL TRESPASS': 12,
        'DANGEROUS DRUGS': 13,
        'DANGEROUS WEAPONS': 14,
        'DISORDERLY CONDUCT': 15,
        'ENDAN WELFARE INCOMP': 16,
        'ESCAPE 3': 17,
        'FELONY ASSAULT': 18,
        'FORGERY': 19,
        'FRAUDS': 20,
        'FRAUDULENT ACCOSTING': 21,
        'GAMBLING': 22,
        'GRAND LARCENY': 23,
        'GRAND LARCENY OF MOTOR VEHICLE': 24,
        'HARRASSMENT 2': 25,
        'HOMICIDE-NEGLIGENT,UNCLASSIFIE': 26,
        'INTOXICATED & IMPAIRED DRIVING': 27,
        'INTOXICATED/IMPAIRED DRIVING': 28,
        'JOSTLING': 29,
        'KIDNAPPING': 30,
        'KIDNAPPING & RELATED OFFENSES': 31,
        'LOITERING/GAMBLING (CARDS, DIC': 32,
        'MISCELLANEOUS PENAL LAW': 33,
        'MURDER & NON-NEGL. MANSLAUGHTER': 34,
        'NEW YORK CITY HEALTH CODE': 35,
        'NYS LAWS-UNCLASSIFIED FELONY': 36,
        'NYS LAWS-UNCLASSIFIED VIOLATION': 37,
        'OFF. AGNST PUB ORD SENSBLTY &': 38,
        'OFFENSES AGAINST PUBLIC ADMINI': 39,
        'OFFENSES AGAINST PUBLIC SAFETY': 40,
        'OFFENSES AGAINST THE PERSON': 41,
        'OFFENSES INVOLVING FRAUD': 42,
        'OFFENSES RELATED TO CHILDREN': 43,
        'OTHER OFFENSES RELATED TO THEF': 44,
        'OTHER STATE LAWS': 45,
        'OTHER STATE LAWS (NON PENAL LA': 46,
        'PETIT LARCENY': 47,
        'PETIT LARCENY OF MOTOR VEHICLE': 48,
        'POSSESSION OF STOLEN PROPERTY': 49,
        'PROSTITUTION & RELATED OFFENSES': 50,
        'RAPE': 51,
        'ROBBERY': 52,
        'SEX CRIMES': 53,
        'THEFT OF SERVICES': 54,
        'THEFT-FRAUD': 55,
        'UNAUTHORIZED USE OF A VEHICLE': 56,
        'VEHICLE AND TRAFFIC LAWS': 57
    }

    patrol_boro_dict = {
        'PATROL BORO BKLYN NORTH': 1,
        'PATROL BORO BKLYN SOUTH': 2,
        'PATROL BORO BRONX': 3,
        'PATROL BORO MAN NORTH': 4,
        'PATROL BORO MAN SOUTH': 5,
        'PATROL BORO QUEENS NORTH': 6,
        'PATROL BORO QUEENS SOUTH': 7,
        'PATROL BORO STATEN ISLAND': 8
    }

    prem_typ_desc_dict = {
        'ABANDONED BUILDING': 1,
        'AIRPORT TERMINAL': 2,
        'ATM': 3,
        'BANK': 4,
        'BAR/NIGHT CLUB': 5,
        'BEAUTY & NAIL SALON': 6,
        'BOOK/CARD': 7,
        'BRIDGE': 8,
        'BUS (NYC TRANSIT)': 9,
        'BUS (OTHER)': 10,
        'BUS STOP': 11,
        'BUS TERMINAL': 12,
        'CANDY STORE': 13,
        'CEMETERY': 14,
        'CHAIN STORE': 15,
        'CHECK CASHING BUSINESS': 16,
        'CHURCH': 17,
        'CLOTHING/BOUTIQUE': 18,
        'COMMERCIAL BUILDING': 19,
        'CONSTRUCTION SITE': 20,
        'DEPARTMENT STORE': 21,
        'DOCTOR/DENTIST OFFICE': 22,
        'DRUG STORE': 23,
        'DRY CLEANER/LAUNDRY': 24,
        'FACTORY/WAREHOUSE': 25,
        'FAST FOOD': 26,
        'FERRY/FERRY TERMINAL': 27,
        'FOOD SUPERMARKET': 28,
        'GAS STATION': 29,
        'GROCERY/BODEGA': 30,
        'GYM/FITNESS FACILITY': 31,
        'HIGHWAY/PARKWAY': 32,
        'HOSPITAL': 33,
        'HOTEL/MOTEL': 34,
        'JEWELRY': 35,
        'LIQUOR STORE': 36,
        'LOAN COMPANY': 37,
        'MAILBOX INSIDE': 38,
        'MAILBOX OUTSIDE': 39,
        'MARINA/PIER': 40,
        'MOSQUE': 41,
        'OPEN AREAS (OPEN LOTS)': 42,
        'OTHER': 43,
        'OTHER HOUSE OF WORSHIP': 44,
        'PARK/PLAYGROUND': 45,
        'PARKING LOT/GARAGE (PRIVATE)': 46,
        'PARKING LOT/GARAGE (PUBLIC)': 47,
        'PHOTO/COPY': 48,
        'PRIVATE/PAROCHIAL SCHOOL': 49,
        'PUBLIC BUILDING': 50,
        'PUBLIC SCHOOL': 51,
        'RESIDENCE - APT. HOUSE': 52,
        'RESIDENCE - PUBLIC HOUSING': 53,
        'RESIDENCE-HOUSE': 54,
        'RESTAURANT/DINER': 55,
        'SHOE': 56,
        'SMALL MERCHANT': 57,
        'SOCIAL CLUB/POLICY': 58,
        'STORAGE FACILITY': 59,
        'STORE UNCLASSIFIED': 60,
        'STREET': 61,
        'SYNAGOGUE': 62,
        'TAXI (LIVERY LICENSED)': 63,
        'TAXI (YELLOW LICENSED)': 64,
        'TAXI/LIVERY (UNLICENSED)': 65,
        'TELECOMM. STORE': 66,
        'TRANSIT - NYC SUBWAY': 67,
        'TRANSIT FACILITY (OTHER)': 68,
        'TRAMWAY': 69,
        'TUNNEL': 70,
        'VARIETY STORE': 71,
        'VIDEO STORE': 72,
    }

    vic_age_group_dict = {
        '-5': 1,
        '-43': 2,
        '-51': 3,
        '-55': 4,
        '-61': 5,
        '-76': 6,
        '-940': 7,
        '-942': 8,
        '-955': 9,
        '-956': 10,
        '-958': 11,
        '-968': 12,
        '-972': 13,
        '-974': 14,
        '18-24': 15,
        '25-44': 16,
        '45-64': 17,
        '65+': 18,
        '922': 19,
        '951': 20,
        '954': 21,
        '970': 22,
        '972': 23,
        '<18': 24,
        'UNKNOWN': 25
    }

    vic_race_dict = {'AMER IND': 1, 'ASIAN/PAC.ISL': 2, 'BLACK': 3, 'BLACK HISPANIC': 4, 'UNKNOWN': 5, 'WHITE': 6, 'WHITE HISPANIC': 7}
    vic_sex_dict = {'D': 1, 'E': 2, 'F': 3, 'M': 4, 'U': 5}

    selected_numeric_columns = raw_data.iloc[:, nypd_num_indices]
    selected_categorical_columns = raw_data.iloc[:, nypd_cat_indices]

    combined_selection = pd.concat([selected_numeric_columns, selected_categorical_columns], axis=1)

    combined_selection['boro_nm'] = combined_selection['boro_nm'].map(boro_nm_dict)  # 2
    combined_selection['boro_nm'] = combined_selection['boro_nm'].fillna(combined_selection['boro_nm'].mode()[0])

    combined_selection['cmplnt_fr_dt'] = combined_selection['cmplnt_fr_dt'].str.replace('-', '').astype(int)  # 3
    combined_selection['cmplnt_fr_tm'] = combined_selection['cmplnt_fr_tm'].str.replace(':', '').astype(int)  # 4

    combined_selection['crm_atpt_cptd_cd'] = combined_selection['crm_atpt_cptd_cd'].map(crm_atpt_cptd_cd_dict)  # 7, ignore for kp
    combined_selection['jurisdiction_code'] = combined_selection['jurisdiction_code'].fillna(combined_selection['jurisdiction_code'].mode()[0])  # 10
    combined_selection['juris_desc'] = combined_selection['juris_desc'].map(juris_desc_dict)  # 11, ignore for kp
    combined_selection['law_cat_cd'] = combined_selection['law_cat_cd'].map(law_cat_cd_dict)  # 13, ignore for kp

    combined_selection['ofns_desc'] = combined_selection['ofns_desc'].map(ofns_desc_dict)  # 15
    combined_selection['ofns_desc'] = combined_selection['ofns_desc'].fillna(combined_selection['ofns_desc'].mode()[0])

    combined_selection['patrol_boro'] = combined_selection['patrol_boro'].map(patrol_boro_dict)  # 17
    combined_selection['patrol_boro'] = combined_selection['patrol_boro'].fillna(combined_selection['patrol_boro'].mode()[0])

    combined_selection['pd_cd'] = combined_selection['pd_cd']  # 18
    combined_selection['pd_cd'] = combined_selection['pd_cd'].fillna(combined_selection['pd_cd'].mean())

    combined_selection['prem_typ_desc'] = combined_selection['prem_typ_desc'].map(prem_typ_desc_dict)  # 20
    combined_selection['prem_typ_desc'] = combined_selection['prem_typ_desc'].fillna(combined_selection['prem_typ_desc'].mode()[0])

    combined_selection['rpt_dt'] = combined_selection['rpt_dt'].str.replace('-', '').astype(int)  # 21

    combined_selection['vic_age_group'] = combined_selection['vic_age_group'].map(vic_age_group_dict)  # 27, ignore for kp
    combined_selection['vic_race'] = combined_selection['vic_race'].map(vic_race_dict)  # 28, ignore for kp
    combined_selection['vic_sex'] = combined_selection['vic_sex'].map(vic_sex_dict)  # 29, ignore for kp

    combined_selection['x_coord_cd'] = combined_selection['x_coord_cd'].fillna(combined_selection['x_coord_cd'].mean())  # 30
    combined_selection['y_coord_cd'] = combined_selection['y_coord_cd'].fillna(combined_selection['y_coord_cd'].mean())  # 31
    combined_selection['latitude'] = combined_selection['latitude'].fillna(combined_selection['latitude'].mean())  # 32
    combined_selection['longitude'] = combined_selection['longitude'].fillna(combined_selection['longitude'].mean())  # 33

    return combined_selection
