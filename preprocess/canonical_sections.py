import re


USE_TYPE_PREFIX = {
    'addendum',
    'assessment',
    'comparison',
    'interpretation',
    'reason',
    'reason_for',
    'service',
    'action',
    'plan',
    'response',
    'conclusions',
    'interpretation',
    'findings',
}

_DEFAULT = {
    'addendum': [
        r'ADDENDUM'
    ],
    'allergies': [
        r'ALLERGY',
        r'ALLERGIES'
    ],
    'approved': [
        (r'APPROVED', 'em')
    ],
    'assessment_plan': [
        r'ASSESSMENT.+PLAN',
    ],
    'attending': [
        (r'ATTENDING', 'em')
    ],
    'chief_complaint': [
        r'(CHIEF|SUBJECTIVE) COMPLAINT'
    ],
    'comparison': [
        r'COMPARISON'
    ],
    'completed_by': [
        (r'(COMPLETED|DICTATED) BY', 'em')
    ],
    'date_time': [
        r'DATE',
        (r'TIME', 'em')
    ],
    'discharge_condition': [
        r'DISCHARGE CONDITION',
        r'CONDITION (ON|AT) DISCHARGE'
    ],
    'discharge_diagnosis': [
        r'DIAGNOS(I|E)S',
        r'\bDX(S)?\b'
    ],
    'discharge_disposition': [
        r'DISPOSITION'
    ],
    'discharge_followup': [
        r'FOLLOW(-)?UP INSTRUCTIONS',
        r'FOLLOW(-)?UP (ON|AT) DISCHARGE'
    ],
    'discharge_instructions': [
        r'DISCHARGE INSTRUCTIONS',
        r'INSTRUCTIONS (ON|AT) DISCHARGE'
    ],
    'discharge_status': [
        r'DISCHARGE STATUS',
        r'STATUS (ON|AT) DISCHARGE'
    ],
    'facility': [
        (r'FACILITY^', 'em')
    ],
    'family_history': [
        r'FAMILY (HISTORY|HX)',
        r'FHX'
    ],
    'findings': [
        r'FINDING(S)?',
        r'IMPRESS(S)?ION',
        r'IMPRESSON'
    ],
    'hospital_course': [
        r'\bCOURSE\b',
    ],
    'hpi': [
        r'HISTORY OF( THE)? PRESENT(ING)? ILLNESS',
        r'\bHPI\b',
        r'HISTORY OF( THE)? PE',
    ],
    'imaging': [
        r'\bRADIOGR',
        r'\bRADIOL',
        r'IMAGES',
        r'IMAGING',
    ],
    'labs': [
        r'RESULTS',
        r'LABS',
        r'LABORATORY( DATA)?'
    ],
    'medications': [
        r'MEDICINE',
        r'MEDICATION(S)?',
        r'\bMED(S)?\b',
        r'OTHER DRUGS',
    ],
    'past_medical_history': [
        r'(PAST )?MEDICAL HISTORY',
        r'CLINICAL HISTORY',
        r'MEDICAL HX',
        r'MHX',
        r'BRIEF (HISTORY|HX)',
        (r'HISTORY', 'em')
    ],
    'patient_info': [
        (r'NAME', 'em'),
        (r'PHONE', 'em'),
        (r'ADDRESS', 'em'),
        (r'LOCATION', 'em'),
        (r'APPOINTMENT', 'em'),
    ],
    'personal_social_history': [
        r'SOCIAL (HISTORY|HX)',
        r'OCCUPATIONAL (HISTORY|HX)',
        r'\bSHX\b',
        r'\bPHX\b',
        r'\bPSHX\b',
    ],
    'physician_doctor': [
        (r'(ATTENDING|PRIMARY CARE|REFERRING) PHYSICIAN', 'em'),
        (r'PHYSICIANS', 'em'),
        (r'PROVIDER', 'em'),
        (r'(ATTENDING|PRIMARY CARE|REFERRING) DOCTOR', 'em'),
        (r'(ATTENDING|PRIMARY CARE|REFERRING) SURGEON', 'em'),
        (r'(ATTENDING|PRIMARY CARE|REFERRING) RADIOLOGIST', 'em'),
    ],
    'physical_exam': [
        r'\bPHYSICAL( EXAM)?'
    ],
    'reason': [
        (r'REASON', 'em'),
        r'REASON FOR( THIS)? EXAM',
    ],
    'reason_for': [
        r'REASON FOR',
    ],
    'review_of_systems': [
        r'REVIEW OF SYSTEM(S)?',
        r'REVIEW OF SYTEMS',
        r'\bROS\b',
    ],
    'service': [
        (r'SERVICE', 'em'),
    ],
    'sex': [
        (r'SEX', 'em')
    ],
    'studies': [
        r'STUDIES'
    ],
    'surgical_procedure': [
        r'MAJOR SURGICAL',
        r'SURGERY',
        r'INVASIVE PROCEDURE',
        r'SURGICAL (HISTORY|HX)'
    ],
}


SECTION_MAPS = {
    'Case Management ': {
        'action': [
            (r'ACTION', 'em')
        ],
        'assessment': [
            (r'ASSESSMENT', 'em')
        ],
        'functional_status_home_family_assessment': [
            r'Functional Status'
        ],
        'discharge_plan': [
            r'DISCHARGE'
        ],
        'narrative': [
            r'NARRATIVE'
        ],
        'plan': [
            (r'PLAN', 'em')
        ],
        'response': [
            (r'RESPONSE', 'em')
        ],
    },
    'Consult': _DEFAULT,
    'Discharge summary': _DEFAULT,
    'Echo': {
        'conclusions': [
            r'CONCLUSION'
        ],
        'findings': [
            r'FINDING'
        ],
        'interpretation': [
            r'INTERPRETATION'
        ],
        'test_information': [
            r'TEST INFORMATION'
        ]
    },
    'ECG': {},
    'General': {},
    'Nursing': {
        'action': [
            r'ACTION'
        ],
        'assessment': [
            r'ASSESSMENT'
        ],
        'plan': [
            r'PLAN'
        ],
        'response': [
            r'RESPONSE'
        ],
    },
    'Nutrition': {},
    'Nursing': {},
    'Nursing/other': {},
    'Pharmacy': {},
    'Physician ': _DEFAULT,
    'Radiology': {
        'addendum': [
          r'ADDENDUM'
        ],
        'bones_soft_tissues': [
          r'BONE.+TISSUE'
        ],
        'bone_windows': [
            r'BONE.+WINDOW'
        ],
        'comparison': [
           r'COMPARISON'
        ],
        'conclusions': [
            r'CONCLUSION(S)?'
        ],
        'condition': [
            r'CONDITION'
        ],
        'ct_abdomen': [
            r'ABD.+CT',
            r'CT.+ABD'
        ],
        'ct_chest': [
            r'CHEST.+CT',
            r'CT.+CHEST'
        ],
        'ct_head': [
            r'HEAD.+CT',
            r'CT.+HEAD'
        ],
        'ct_pelvis': [
            r'CT.+PELVI',
            r'PELVI.+CT'
        ],
        'ct_other': [
            r'\bCT\b'
        ],
        'diagnosis': [
            r'DIAGNOSIS'
        ],
        'field_of_view': [
          r'FIELD.+VIEW'
        ],
        'findings': [
            r'FINDINGS'
        ],
        'history': [
            r'HISTORY',
            r'CLINICAL INFORMATION'
        ],
        'impression': [
            r'IMPRESSION'
        ],
        'indication': [
           r'INDICATION'
        ],
        'interpretation': [
            (r'INTERPRETATION', 'em')
        ],
        'pfi': [
          r'PFI'
        ],
        'procedure': [
            (r'PROCEDURE(S)?( DETAILS)?', 'em')
        ],
        'radiograph_chest': [
            r'CHEST RADIOGRAPH',
            r'(PORTABLE|AP) CHEST'
        ],
        'radiograph_other': [
            r'RADIOGRAPH'
        ],
        'reason': [
            r'REASON'
        ],
        'study': [
            (r'STUDY', 'em')
        ],
        'technique': [
            r'TECHNIQUE'
        ],
        'views': [
            r'VIEWS'
        ],
        'wet_read': [
            r'WET READ'
        ],
        'xray_chest': [
            r'CHEST.+X(-| )?RAY',
            r'X(-| )?RAY.+CHEST'
        ],
        'xray_other': [
            r'X(-| )?RAY'
        ],
    },
    'Rehab Services': {
        'activity_orders': [
            r'ACTIVITY ORDERS'
        ],
        'arousal_attention_cognition_communication': [
            r'AROUSAL.+ATTENTION.+COGNITION.+COMMUNICATION'
        ],
        'diagnosis': [
            r'DIAGNOS(I|E)S',
            r'\bDX(S)?\b'
        ],
        'evaluation': [
            (r'EVALUATION', 'em')
        ],
        'hpi': [
            r'HISTORY OF( THE)? PRESENT(ING)? ILLNESS',
            r'\bHPI\b',
            r'HISTORY OF( THE)? PE',
            (r'HISTORY', 'em')
        ],
        'labs': [
            r'LABS'
        ],
        'living_environment': [
            r'LIVING ENVIRONMENT'
        ],
        'oral_motor_exam': [
            r'ORAL MOTOR EXAM'
        ],
        'past_medical_history': [
            r'(PAST )?MEDICAL HISTORY',
            r'CLINICAL HISTORY',
            r'MEDICAL HX',
            r'MHX',
            r'BRIEF (HISTORY|HX)',
            (r'HISTORY', 'em')
        ],
        'radiology': [
            (r'RADIOLOGY', 'em')
        ],
        'recommendations': [
            r'RECOMMENDATIONS'
        ],
        'reason': [
           r'REASON OF REFERRAL'
        ],
        'personal_social_history': [
            r'SOCIAL (HISTORY|HX)',
            r'OCCUPATIONAL (HISTORY|HX)',
            r'\bSHX\b',
            r'\bPHX\b',
            r'\bPSHX\b',
        ],
        'swallowing_assessment': [
            r'SWALLOWING (ASSESSMENT|EVAL)'
        ],
        'summary_impression': [
            r'IMPRESSION'
        ],
        'surgical_history': [
            r'SURGICAL (HISTORY|HX)'
        ],
    },
    'Respiratory ': {},
    'Social Work': {}
}


def _to_snake(str):
    return re.sub(r'[/\s]+', '_', str.strip())


def get_canonical_section(section, category):
    section = re.sub(r'\/', ' ', section).upper()
    for canonical, patterns in SECTION_MAPS[category].items():
        for pattern in patterns:
            if type(pattern) == tuple:
                if pattern[1] == 'em':
                    if re.match(r'^' + pattern[0] + r'$', section, flags=re.IGNORECASE):
                        category = _to_snake(category) if canonical in USE_TYPE_PREFIX else 'CANONICAL'
                        return (category + '->' + canonical).upper()
                else:
                    raise Exception('Invalid pattern match type!')
            else:
                if re.match(r'^(.*?)' + pattern, section, flags=re.IGNORECASE):
                    category = _to_snake(category) if canonical in USE_TYPE_PREFIX else 'CANONICAL'
                    return (category + '->' + canonical).upper()
    medication_regex = r'^(.*?)\b(CAPSULE|SIG|TABLET|MG|AEROSOL|SIG)\b'
    if re.match(medication_regex, section):
        return 'MEDICATION_DETAIL'
    return section

