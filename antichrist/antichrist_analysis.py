#!/usr/bin/env python3
"""
Antichrist Bayesian Analysis Tool

This script performs a comprehensive Bayesian analysis to evaluate how well
different historical figures match biblical prophecies about the Antichrist.
It uses structured YAML data for both prophecies and subjects.

Usage:
  python antichrist_analysis.py [--prior PRIOR] [--subjects SUBJECTS...]

Options:
  --prior PRIOR        Prior probability (default: 1e-6)
  --subjects SUBJECTS  List of subjects to analyze (default: all)
"""

import argparse
import math
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class AntichristAnalysis:
    def __init__(self, data_dir="data", prior_probability=1e-6):
        """Initialize the analysis tool with data directory and prior probability."""
        self.data_dir = Path(data_dir)
        self.prior = prior_probability
        self.prophecies = {}
        self.subjects = {}
        self.results = {}

        # Load all data
        self.load_prophecies()
        self.load_subjects()

    def load_prophecies(self):
        """Load all prophecies from YAML files in the prophecies directory."""
        prophecy_dir = self.data_dir / "prophecies"
        prophecy_files = prophecy_dir.glob("*.yaml")

        for file_path in prophecy_files:
            with open(file_path, 'r') as f:
                prophecies = yaml.safe_load(f)

                for prophecy in prophecies:
                    prophecy_id = prophecy['id']
                    self.prophecies[prophecy_id] = prophecy

        print(f"Loaded {len(self.prophecies)} prophecies")

    def load_subjects(self):
        """Load all subject data from YAML files in the subjects directory."""
        subject_dir = self.data_dir / "subjects"
        subject_files = subject_dir.glob("*.yaml")

        for file_path in subject_files:
            with open(file_path, 'r') as f:
                subject_data = yaml.safe_load(f)
                subject_name = subject_data['name']
                self.subjects[subject_name] = subject_data

        print(f"Loaded data for {len(self.subjects)} subjects")

    def analyze_subject(self, subject_name):
        """Perform Bayesian analysis for a specific subject."""
        if subject_name not in self.subjects:
            print(f"Subject '{subject_name}' not found")
            return None

        subject = self.subjects[subject_name]
        ratings = subject['prophecy_ratings']

        # Count prophecies with ratings
        matched_prophecies = len(ratings)
        total_prophecies = len(self.prophecies)
        coverage = matched_prophecies / total_prophecies

        # Calculate prior odds
        prior_odds = self.prior / (1 - self.prior)

        # For each prophecy with a rating, calculate likelihood ratio
        # Likelihood ratio = P(Evidence|Hypothesis) / P(Evidence|~Hypothesis)
        # where P(Evidence|Hypothesis) is the rating value
        # and P(Evidence|~Hypothesis) is 0.1 for simplicity (can be refined)
        log_likelihood_ratio_sum = 0

        prophecy_lrs = {}  # Store LRs for each prophecy to identify strongest matches

        for prophecy_id, rating in ratings.items():
            # Get the prophecy Bayesian values if available
            if prophecy_id in self.prophecies and 'bayesian' in self.prophecies[prophecy_id]:
                bayesian = self.prophecies[prophecy_id]['bayesian']
                base_p_h = bayesian['P_H']
                base_p_not_h = bayesian['P_notH']
            else:
                # Default values if Bayesian values not found
                base_p_h = 0.8
                base_p_not_h = 0.2

            # Handle rating of 0 as evidence against the hypothesis
            if rating == 0:
                # A rating of 0 means the prophecy is not fulfilled

                # For the Antichrist hypothesis (H), if they truly are the Antichrist,
                # they should fulfill most prophecies, so not fulfilling one is evidence against
                # However, some prophecies may not apply to all historical periods
                p_evidence_h = 0.05  # Small probability that Antichrist wouldn't fulfill a prophecy

                # For the non-Antichrist hypothesis (~H), we'd expect most people
                # NOT to fulfill most of the prophecies
                p_evidence_not_h = 0.70  # High probability for non-Antichrist to not fulfill prophecy

                # Calculate evidence ratio - this is a value < 1
                lr = p_evidence_h / p_evidence_not_h
            else:
                # Normal case for non-zero ratings
                # Scale the P_H by the subject's fulfillment rating
                p_evidence_h = base_p_h * rating

                # For P_notH, use a less aggressive scaling
                p_evidence_not_h = base_p_not_h + ((1.0 - base_p_not_h) * (1.0 - rating) * 0.5)

                # Ensure P_notH is never 0 to avoid division by zero
                p_evidence_not_h = max(0.001, p_evidence_not_h)

                # Calculate likelihood ratio without artificial caps
                # This allows extraordinary evidence to have its full impact
                lr = p_evidence_h / p_evidence_not_h
            prophecy_lrs[prophecy_id] = lr

            # Use logarithm to avoid numerical overflow
            log_likelihood_ratio_sum += math.log10(lr)

        # Calculate posterior odds using logarithms
        log_posterior_odds = math.log10(prior_odds) + log_likelihood_ratio_sum
        posterior_odds = 10 ** log_posterior_odds

        # Convert to probability
        posterior_probability = posterior_odds / (1 + posterior_odds)

        # Sort prophecies by LR to find strongest matches
        top_prophecies = sorted(prophecy_lrs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Store results
        self.results[subject_name] = {
            'prior_probability': self.prior,
            'prophecy_coverage': coverage,
            'log_likelihood_ratio': log_likelihood_ratio_sum,
            'posterior_probability': posterior_probability,
            'top_prophecies': top_prophecies
        }

        return self.results[subject_name]

    def analyze_all_subjects(self):
        """Analyze all loaded subjects."""
        for subject_name in self.subjects:
            self.analyze_subject(subject_name)

        return self.results

    def print_results(self, subject_name=None):
        """Print analysis results for a subject or all subjects."""
        if subject_name:
            if subject_name not in self.results:
                print(f"No results for '{subject_name}'")
                return

            self._print_subject_result(subject_name)
        else:
            # Print all results, sorted by posterior probability
            sorted_subjects = sorted(
                self.results.items(),
                key=lambda x: x[1]['posterior_probability'],
                reverse=True
            )

            print("\n=== OVERALL RESULTS ===")
            print("Subject                                    | Posterior Probability | Log Likelihood Ratio")
            print("-"*90)

            for name, result in sorted_subjects:
                prob = result['posterior_probability']
                llr = result['log_likelihood_ratio']

                # Format probability based on its value
                if prob > 0.1:
                    prob_str = f"{prob:.0%}"
                elif prob > 0.01:
                    prob_str = f"{prob:.1%}"
                else:
                    prob_str = f"{prob:.2%}"

                print(f"{name:40} | {prob_str:20} | {llr:.1f}")

            print("\nDetailed results for each subject:")
            for name, _ in sorted_subjects:
                self._print_subject_result(name)

    def _print_subject_result(self, subject_name):
        """Print detailed results for a specific subject."""
        result = self.results[subject_name]
        subject = self.subjects[subject_name]

        # Format probability based on its value
        prob = result['posterior_probability']
        if prob > 0.1:
            prob_str = f"{prob:.0%}"
        elif prob > 0.01:
            prob_str = f"{prob:.1%}"
        else:
            prob_str = f"{prob:.2%}"

        print(f"\n=== {subject_name} ({subject['title']}, {subject['time_period']}) ===")
        print(f"Prior Probability: {self.prior:.6%}")
        print(f"Prophecy Coverage: {result['prophecy_coverage']:.0%} of all prophecies")
        print(f"Log10 Likelihood Ratio: {result['log_likelihood_ratio']:.1f}")
        print(f"This means the evidence is {10**result['log_likelihood_ratio']:.0e} times more likely if {subject_name} is the Antichrist")
        print(f"Posterior Probability: {prob_str}")

        print("\nTop 10 matching prophecies:")
        for prophecy_id, lr in result['top_prophecies']:
            if prophecy_id in self.prophecies:
                prophecy = self.prophecies[prophecy_id]
                print(f"  - {prophecy['reference']} (LR: {lr:.2f}): {prophecy['description']}")

                # Print evidence if available
                if 'evidence' in subject and prophecy_id in subject['evidence']:
                    print(f"    Evidence: {subject['evidence'][prophecy_id]}")
            else:
                print(f"  - {prophecy_id} (LR: {lr:.2f}): [Prophecy details not found]")

    def plot_comparison(self):
        """Generate comparative plots of the results."""
        if not self.results:
            print("No results to plot")
            return

        # Sort subjects by probability
        sorted_subjects = sorted(
            self.results.items(),
            key=lambda x: x[1]['posterior_probability'],
            reverse=True
        )

        names = [name for name, _ in sorted_subjects]
        probabilities = [result['posterior_probability'] for _, result in sorted_subjects]
        log_lrs = [result['log_likelihood_ratio'] for _, result in sorted_subjects]

        # Plot 1: Posterior probabilities (bar chart)
        plt.figure(figsize=(12, 8))
        plt.bar(names, probabilities)
        plt.title('Posterior Probability of Being the Antichrist', pad=20, fontsize=14)
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1.1)  # Higher ylim to make room for labels
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add rounded probability values on top of bars
        for i, v in enumerate(probabilities):
            if v > 0.99:  # For very high probabilities
                plt.text(i, v + 0.05, f"{v:.0%}", ha='center', fontsize=10)
            elif v > 0.1:
                plt.text(i, v + 0.02, f"{v:.0%}", ha='center', fontsize=10)
            elif v > 0.01:
                plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10)
            else:
                plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('antichrist_probability.png')
        print("Probability comparison saved as 'antichrist_probability.png'")

        # Plot 2: Log likelihood ratios (bar chart)
        plt.figure(figsize=(12, 8))
        plt.bar(names, log_lrs)
        plt.title('Log10 Likelihood Ratio (Evidence Strength)', fontsize=14)
        plt.ylabel('Log10(LR)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add LR values on top of bars with 1 decimal place
        for i, v in enumerate(log_lrs):
            plt.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('antichrist_likelihood.png')
        print("Likelihood ratio comparison saved as 'antichrist_likelihood.png'")

        # Generate a heat map of prophecy fulfillment
        self._plot_prophecy_heatmap()

    def _plot_prophecy_heatmap(self):
        """Generate a heatmap showing prophecy fulfillment across subjects."""
        if not self.subjects or not self.results:
            return

        # Get all prophecy IDs
        prophecy_ids = list(self.prophecies.keys())

        # Sort subjects by posterior probability (highest first)
        sorted_subjects = sorted(
            self.results.items(),
            key=lambda x: x[1]['posterior_probability'],
            reverse=True
        )

        # Extract just the subject names in order
        subject_names = [name for name, _ in sorted_subjects]

        # Set up the plot
        plt.figure(figsize=(20, 12))

        # Group prophecy IDs by book
        book_groups = {}
        for prophecy_id in prophecy_ids:
            book = prophecy_id.split('_')[0]  # Extract book prefix
            if book not in book_groups:
                book_groups[book] = []
            book_groups[book].append(prophecy_id)

        # Sort prophecies by book
        sorted_prophecies = []
        for book in sorted(book_groups.keys()):
            sorted_prophecies.extend(sorted(book_groups[book]))

        # Create the sorted matrix directly using the sorted subject order
        sorted_matrix = np.zeros((len(subject_names), len(sorted_prophecies)))

        # Fill the sorted matrix
        for i, subject_name in enumerate(subject_names):
            subject = self.subjects[subject_name]
            ratings = subject['prophecy_ratings']

            for j, prophecy_id in enumerate(sorted_prophecies):
                if prophecy_id in ratings:
                    sorted_matrix[i, j] = ratings[prophecy_id]

        # Create the heatmap with the rows in descending order of probability
        plt.imshow(sorted_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Prophecy Fulfillment Rating')

        # Format labels with posterior probabilities
        subject_labels = []
        for i, (name, result) in enumerate(sorted_subjects):
            prob = result['posterior_probability']
            if prob > 0.1:
                prob_str = f"{prob:.0%}"
            elif prob > 0.01:
                prob_str = f"{prob:.1%}"
            elif prob > 0.0001:
                prob_str = f"{prob:.2%}"
            else:
                prob_str = f"~0%"
            subject_labels.append(f"{name} ({prob_str})")

        # Set the axis ticks and labels
        plt.yticks(range(len(subject_names)), subject_labels)
        plt.xticks(range(len(sorted_prophecies)), sorted_prophecies, rotation=90)

        plt.title('Prophecy Fulfillment Ratings Across Subjects (Sorted by Posterior Probability)', fontsize=14)
        plt.tight_layout()
        plt.savefig('prophecy_heatmap.png')
        print("Prophecy heatmap saved as 'prophecy_heatmap.png'")

def main():
    parser = argparse.ArgumentParser(description='Antichrist Bayesian Analysis Tool')
    parser.add_argument('--prior', type=float, default=1e-6,
                        help='Prior probability (default: 1e-6, one in a million)')
    parser.add_argument('--subjects', nargs='+',
                        help='List of subjects to analyze (default: all)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')

    args = parser.parse_args()

    # Create and run the analysis
    analysis = AntichristAnalysis(prior_probability=args.prior)

    if args.subjects:
        for subject in args.subjects:
            analysis.analyze_subject(subject)
            analysis.print_results(subject)
    else:
        analysis.analyze_all_subjects()
        analysis.print_results()

    if args.plot:
        analysis.plot_comparison()

if __name__ == "__main__":
    main()
