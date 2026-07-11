from agent_core.security.pii import mask_pii


def test_masks_email():
    assert mask_pii("contact me at rahul@example.com please") == "contact me at [EMAIL] please"


def test_masks_10_digit_phone():
    assert "[PHONE]" in mask_pii("call me at 9876543210 tomorrow")


def test_masks_12_digit_id_number():
    assert "[ID_NUMBER]" in mask_pii("my aadhaar is 1234 5678 9012")


def test_masks_card_number_before_id_pattern_would_grab_part_of_it():
    masked = mask_pii("card 4111111111111111 expires soon")
    assert "[CARD_NUMBER]" in masked
    assert "4111" not in masked


def test_leaves_non_pii_text_alone():
    assert mask_pii("book a flight to Mumbai tomorrow") == "book a flight to Mumbai tomorrow"
