"""Module containing constants used in the application."""

OUTPUT_FILE = "output_dataframe\\redundancies.csv"

IRRELVANT_COLS = [
    "report_id",
    "report_ref_counter",
    "report_reference",
    "report_number",
    "report_submission_date",
    "report_elise_ref",
    "report_elise_doc_size",
    "governorate",
    "vehicles_type",
    "vehicles_model",
    "vehicles_defendant",
    "vehicles_manufacturer",
    "casualties_address",
    "casualties_last_name",
    "casualties_birth_date",
    "casualties_death_date",
    "casualties_death_time",
    "casualties_first_name",
    "casualties_issue_date",
    "casualties_death_place",
    "casualties_death_governorate",
    "casualties_casualty_id",
    "casualties_death_medical_cause",
    "casualties_registration_number",
    "vehicles_vehicle_id",
]

NUMERICAL_COLS = [
    "vehicles_market_share",
    "casualties_age",
    "casualties_gender",
    "casualties_casualty_type",
    "casualties_casualty_category",
]

CATEGORICAL_COLS = [
    "region",
    "accident_place",
    "accident_time",
    "vehicles_foreign_insurance",
    "casualties_social_state",
    "casualties_health_institution",
    "vehicles_plate_number",
    "vehicles_insurance_name",
]

DATE_COLS = ["report_accident_date", "report_date"]

NUMERICAL_COLS_AFTER_PIPELINE = [
    "num__vehicles_market_share",
    "num__casualties_age",
    "num__casualties_gender",
    "num__casualties_casualty_type",
    "num__casualties_casualty_category",
    "remainder__casualties_identity",
    "remainder__is_police",
    "remainder__report_accident_date_year",
    "remainder__report_accident_date_month",
    "remainder__report_accident_date_day",
    "remainder__report_date_year",
    "remainder__report_date_month",
    "remainder__report_date_day",
]

CAT_COLS_AFTER_IMPUTING = [
    "cat__region",
    "cat__accident_place",
    "cat__accident_time",
    "cat__vehicles_foreign_insurance",
    "cat__casualties_social_state",
    "cat__casualties_health_institution",
    "cat__vehicles_plate_number",
    "cat__vehicles_insurance_name",
]

HIGH_CARDINALITY_COLS = ["cat__accident_place", "cat__vehicles_plate_number"]

LOW_CARDINALITY_COLS = [
    col for col in CAT_COLS_AFTER_IMPUTING if col not in HIGH_CARDINALITY_COLS
]

PLATE_NUMBER_FEATURES = [
    "plate_length",
    "plate_num_digits",
    "plate_num_alpha",
    "plate_has_hyphen",
    "plate_starts_with_alpha",
    "plate_alpha_numeric_segments",
    "plate_prefix",
    "plate_suffix",
    "plate_number_hash",
    "plate_digit_segments",
    "plate_alpha_segments",
    "plate_detailed_prefix",
    "plate_detailed_suffix",
    "plate_numeric_pattern",
    "plate_alpha_pattern",
]
