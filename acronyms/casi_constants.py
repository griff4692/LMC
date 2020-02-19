# List of constants used for preprocessing the CASI dataset.  Reasons for removals include:
# Duplicate LFs, LFs that don't actually correspond to SF or are just word senses of SF (same surface form)
# These are not essential transformations - they just clean the dataset in an ad hoc fashion.

LF_BLACKLIST = {
    'type A, type B',
    'American Society of Anaesthesiologists;American Society of Anesthesiologists',
    'California',
    'Iowa',
    'Fairview Southdale Hospital',
    'Los Angeles',
    'Louisiana;louisiana;Louisiana (geographic location)',
    'physician associates',
    'right',
}

SF_BLACKLIST = {
    'C3',
    'C4',
    'T1',
    'T2',
    'T3',
    'T4'
}

LF_MAPPING = {
    'physician associates': ("Assistant, Physicians';Assistants, Physician;Assistants, Physicians';PHYSICIAN ASSISTANT;"
                             "Physician Assistant;Physician Assistants;Physician's Assistants;Physicians Assistants;"
                             "Physicians' Assistant;Physicians' Assistants;Physicians' assistants;physician assistant;"
                             "physician assistants;physician's assistant;physician's assistants"),
    'gutta': 'guttae'
}
