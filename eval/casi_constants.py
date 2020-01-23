LF_BLACKLIST = set([
    'type A, type B',
    'American Society of Anaesthesiologists;American Society of Anesthesiologists',
    'California',
    'Iowa',
    'Fairview Southdale Hospital',
    'Los Angeles',
    'Louisiana;louisiana;Louisiana (geographic location)',
    'physician associates',
    'right',
])

SF_BLACKLIST = set([
    'C3',
    'C4',
    'T1',
    'T2',
    'T3',
    'T4'
])

LF_MAPPING = {
    'physician associates': ("Assistant, Physicians';Assistants, Physician;Assistants, Physicians';PHYSICIAN ASSISTANT;"
                             "Physician Assistant;Physician Assistants;Physician's Assistants;Physicians Assistants;"
                             "Physicians' Assistant;Physicians' Assistants;Physicians' assistants;physician assistant;"
                             "physician assistants;physician's assistant;physician's assistants"),
    'gutta': 'guttae'
}