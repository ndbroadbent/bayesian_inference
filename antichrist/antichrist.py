# See: Could American Evangelicals Spot the Antichrist? Here Are the Biblical Predictions:
# https://www.benjaminlcorey.com/could-american-evangelicals-spot-the-antichrist-heres-the-biblical-predictions/
#
# Bayesian Analysis for All Prophesies
# -------------------------------------
# This script assumes that each prophecy provides independent evidence.
# For each prophecy, we define:
#   - P_H: Probability of observing this evidence if Trump is the Antichrist.
#   - P_notH: Probability of observing this evidence if Trump is NOT the Antichrist.
#
# The likelihood ratio for each piece of evidence is LR = P_H / P_notH.
# We assume a very skeptical prior probability P(H) = 0.000001 (one in a million chance) that Trump is the Antichrist.
# Then, the posterior odds are given by:
#   posterior_odds = (P(H)/(1-P(H))) * (Product over all evidence of LR)
# And the final posterior probability is:
#   P(H|E) = posterior_odds / (1 + posterior_odds)
#
# The following evidence items come from the Benjamin Corey article and bible references:
# There are 37 prophecies in total derived from the article.
#
# Note: These numbers are subjective estimates; adjusting them will affect the final probability.

import math  # For handling logarithms to avoid numerical overflow

config = {
    # 1. Military Superpower Nation (Daniel 7:23)
    "military_superpower": {"P_H": 0.9, "P_notH": 0.5},  # US is a superpower; if AC, high chance

    # 2. Boastful Speeches (Daniel 7:8, Revelation 13:5)
    "boastful_speeches": {"P_H": 0.9, "P_notH": 0.4},  # Excessive boasting is characteristic

    # 3. Public Threats (Revelation 13:2, Daniel 7:4)
    "public_threats": {"P_H": 0.85, "P_notH": 0.3},  # "Roaring" like a lion with threats

    # 4. Obsessed with Winning (Revelation 6:2)
    "obsessed_with_winning": {"P_H": 0.9, "P_notH": 0.4},  # Unusually focused on conquering/winning

    # 5. Seven Hills/Towers (Revelation 13:1, Ch. 17)
    "seven_towers": {"P_H": 0.8, "P_notH": 0.1},  # 7 buildings with his name is quite specific

    # 6. Global Following (Revelation 13:3)
    "global_following": {"P_H": 0.85, "P_notH": 0.3},  # Worldwide attention and following

    # 7. Political Outsider (Daniel 11:21)
    "political_outsider": {"P_H": 0.9, "P_notH": 0.3},  # Non-political background winning unexpectedly

    # 8. Great and Greater Things (Daniel 7:20)
    "great_greater_things": {"P_H": 0.95, "P_notH": 0.1},  # Specific language match to MAGA

    # 9. Collusion and Minority Support (Daniel 11:23)
    "collusion_minority_support": {"P_H": 0.85, "P_notH": 0.2},  # Outside alliance helping win

    # 10. Miraculous Rise to Power (2 Thessalonians 2:9)
    "miraculous_rise": {"P_H": 0.8, "P_notH": 0.3},  # Unexpected victory seen as miraculous

    # 11. Self-Enrichment Through Power (Daniel 11:24)
    "self_enrichment": {"P_H": 0.9, "P_notH": 0.3},  # Using presidency for personal wealth

    # 12. Wanting to Change Times and Laws (Daniel 7:25)
    "change_times_laws": {"P_H": 0.8, "P_notH": 0.4},  # Attempting to alter election timing/rules

    # 13. Champion of Fake News and Deceit (Daniel 8:25, 2 Thessalonians 2:10)
    "fake_news_deceit": {"P_H": 0.95, "P_notH": 0.4},  # Personifying lies and calling truth "fake"

    # 14. Rewarding Loyalty with Power and Real Estate (Daniel 11:39b)
    "rewarding_loyalists": {"P_H": 0.85, "P_notH": 0.3},  # Giving power and deals to loyal followers

    # 15. Blasphemous Public Speech (Revelation 13:5-6)
    "blasphemous_speech": {"P_H": 0.8, "P_notH": 0.2},  # Using God's name inappropriately

    # 16. Christian Support Despite Warning Signs (Matthew 24:24, 2 Thessalonians 2:10)
    "christian_support": {"P_H": 0.9, "P_notH": 0.3},  # Strong evangelical support despite issues

    # 17. Considers Himself Above the Law (2 Thessalonians 2)
    "above_the_law": {"P_H": 0.9, "P_notH": 0.2},  # The "lawless one" immune to consequences

    # 18. Feud with the Southern Border Nation (Daniel 11:40)
    "feud_southern_border": {"P_H": 0.9, "P_notH": 0.3},  # Conflict with Mexico specifically

    # 19. Intentional Harm to Southern Ethnic Group (Daniel 11:28, Zechariah 11:16-17)
    "harm_southern_group": {"P_H": 0.85, "P_notH": 0.2},  # Border policies harming migrants

    # 20. Raging at Reports from the East and North (Daniel 11:44)
    "raging_reports": {"P_H": 0.85, "P_notH": 0.2},  # Anger at negative media/investigations

    # 21. Surviving a Fatal Wound (Revelation 13:3)
    "fatal_wound_recovery": {"P_H": 0.95, "P_notH": 0.005},  # Actually shot in the head and survived

    # 22. Religious Leader Support (Revelation 13:11-12)
    "religious_leader_support": {"P_H": 0.9, "P_notH": 0.3},  # Prominent evangelical endorsements

    # 23. Christians Hold Power but Are Not the Good Guys (Daniel 12:7)
    "christians_power_not_good": {"P_H": 0.85, "P_notH": 0.2},  # Christian political power during reign

    # 24. Alliance with War Criminal in Israel (Daniel 9:26-27)
    "israel_war_criminal_alliance": {"P_H": 0.8, "P_notH": 0.3},  # Support for controversial Israeli actions

    # 25. Focus on Coastal Areas in Israel (Daniel 11:16-18)
    "israel_coastal_focus": {"P_H": 0.75, "P_notH": 0.2},  # Specific interest in Gaza/coastal areas

    # 26. Global Pandemic (Revelation 8:10)
    "global_pandemic": {"P_H": 0.7, "P_notH": 0.3},  # COVID-19 during his presidency

    # 27. Neglect of the Vulnerable During Crisis (Zechariah 11:6)
    "neglect_vulnerable": {"P_H": 0.85, "P_notH": 0.3},  # Poor pandemic response prioritizing economy

    # 28. Strange Dreams Phenomenon (Acts 2:17)
    "strange_dreams": {"P_H": 0.7, "P_notH": 0.3},  # Reports of vivid dreams during pandemic

    # 29. Food Shortages and Price Increases with Cheap Oil (Revelation 6:6)
    "food_shortages_cheap_oil": {"P_H": 0.8, "P_notH": 0.2},  # Pandemic supply chains and oil crash

    # 30. Civil Unrest and Resistance (Daniel 11:33)
    "civil_unrest": {"P_H": 0.85, "P_notH": 0.3},  # BLM protests and resistance movement

    # 31. Church Photo Op After Clearing Protesters (Daniel 11:29-33)
    "church_photo_op": {"P_H": 0.9, "P_notH": 0.05},  # Highly specific incident with Bible at church

    # 32. Violent Rebellion by Supporters (Daniel 11:14, 1 Timothy 3:2)
    "violent_rebellion": {"P_H": 0.9, "P_notH": 0.1},  # Jan 6 insurrection fitting prophecy

    # 33. Putting His Name Above God (Daniel 11:36)
    "name_above_god": {"P_H": 0.8, "P_notH": 0.1},  # Trump Bible with his name at top

    # 34. Mass Disappearance of Agricultural Workers (Matthew 24:40-41)
    "agricultural_workers_disappear": {"P_H": 0.7, "P_notH": 0.3},  # Immigration enforcement/deportations

    # 35. Self-Exaltation as King of Israel (2 Thessalonians 2:4, Revelation 17)
    "king_of_israel": {"P_H": 0.9, "P_notH": 0.05},  # Actually called himself "King of Israel"

    # 36. Followers Wearing a Mark on Their Foreheads (Revelation 13)
    "mark_on_foreheads": {"P_H": 0.85, "P_notH": 0.1},  # MAGA hats as the "mark"

    # 37. Resistance Movement Against Him (Daniel 11:32)
    "resistance_movement": {"P_H": 0.8, "P_notH": 0.4},  # Actual "Resist" movement formed
}

# Set a skeptical but scientifically reasonable prior probability:
prior = 1e-6  # One in a million (10^-6) is still very skeptical but more standard for scientific hypotheses
# Compute prior odds:
prior_odds = prior / (1 - prior)

likelihood_ratio_product = 1.0
log_likelihood_ratio_product = 0.0

print()
print("Prior Probability: {:.6%}".format(prior))
print("Prior Odds: {:.8f}".format(prior_odds))
print("--------------------------------")
print()

print("Individual Likelihood Ratios:")
for key, vals in config.items():
    P_H = vals["P_H"]
    P_notH = vals["P_notH"]
    LR = P_H / P_notH  # Likelihood Ratio for this prophecy
    likelihood_ratio_product *= LR
    log_likelihood_ratio_product += math.log10(LR)  # Use logarithm to avoid numerical overflow
    print(f"  {key}: LR = {LR:.3f}")

print("\nTotal Log10 Likelihood Ratio: {:.2f}".format(log_likelihood_ratio_product))
print("This means the evidence is {:.2e} times more likely if Trump is the Antichrist".format(10**log_likelihood_ratio_product))

# Compute log posterior odds to avoid numerical overflow
log_posterior_odds = math.log10(prior_odds) + log_likelihood_ratio_product
posterior_odds = 10**log_posterior_odds

# Convert odds to probability:
posterior_probability = posterior_odds / (1 + posterior_odds)

print("\nLog10 Posterior Odds: {:.2f}".format(log_posterior_odds))
print("Posterior Odds: {:.2e}".format(posterior_odds))
print("--------------------------------")
print()
print("Posterior Probability that Trump is the Antichrist: {:.10%}".format(posterior_probability))
