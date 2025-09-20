#!/usr/bin/env python3

# Verify the pattern in pseudoseed
data = [
    ("Voucher first", 0.24901819449170001, 0.49803638898340002),
    ("shop_pack1 first", 0.38681086553959998, None),
    ("shop_pack1 second", 0.2342094632665, None),
]

print("=== Verifying pattern ===\n")

for name, returned, stored in data:
    if stored:
        ratio = returned / stored
        print(f"{name}:")
        print(f"  Returned: {returned:.17f}")
        print(f"  Stored:   {stored:.17f}")
        print(f"  Ratio:    {ratio:.17f}")
        print(f"  Is half?  {abs(ratio - 0.5) < 1e-10}")
        print()

# Check if we can reverse engineer the stored values
print("=== Reverse engineering stored values ===\n")

for name, returned, _ in data:
    predicted_stored = returned * 2
    print(f"{name}:")
    print(f"  Returned:         {returned:.17f}")
    print(f"  Predicted stored: {predicted_stored:.17f}")
    print()

# Now let's understand the sequence
print("=== Understanding the sequence ===\n")

# The stored value for Voucher is 0.49803638898340002
# If we apply LCG: next = (2.134453429141 + 0.49803638898340002 * 1.72431234) % 1
voucher_stored = 0.49803638898340002
LCG_A = 2.134453429141
LCG_B = 1.72431234

next_state = (LCG_A + voucher_stored * LCG_B) % 1
print(f"Voucher stored state: {voucher_stored:.17f}")
print(f"After LCG: {next_state:.17f}")
print(f"Returned (half): {next_state / 2:.17f}")
print(f"Expected: 0.49661186021705001")
print(f"Match? {abs(next_state / 2 - 0.49661186021705001) < 1e-10}")