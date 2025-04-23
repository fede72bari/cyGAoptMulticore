#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <algorithm>
#include <optional>
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

// FITNESS PURAMENTE C++
/*
double evaluate_fitness(const std::vector<double>& individual) {
    double sum = 0.0;
    for (double val : individual) {
        sum += val * val;
    }
    return -sum;  // esempio: funzione da minimizzare
}
*/

double evaluate_fitness(const std::vector<double>& individual,
                        const std::vector<double>& reference_signal,
                        int n_cycles,
                        int start_rebuild_index,
                        bool period_related_rebuild_range,
                        double period_multiplier)

{

    int offset_amp = 0;
    int offset_freq = n_cycles;
    int offset_phase = 2 * n_cycles;

    std::vector<double> amps(individual.begin() + offset_amp, individual.begin() + offset_amp + n_cycles);
    std::vector<double> freqs(individual.begin() + offset_freq, individual.begin() + offset_freq + n_cycles);
    std::vector<double> phases(individual.begin() + offset_phase, individual.begin() + offset_phase + n_cycles);

    size_t len = reference_signal.size();
    std::vector<double> signal(len, 0.0);

    for (int c = 0; c < n_cycles; ++c) {
        double A = amps[c];
        double f = freqs[c];
        double p = phases[c];
        for (size_t t = 0; t < len; ++t) {
            signal[t] += A * std::sin(2 * M_PI * f * t + p);
        }
    }

    double mse = 0.0;
	
	size_t compare_start = 0;

	if (period_related_rebuild_range) {
		double min_freq = *std::min_element(freqs.begin(), freqs.end());
		double period = 1.0 / min_freq;
		compare_start = len - static_cast<size_t>(period * period_multiplier);
		if (compare_start < static_cast<size_t>(start_rebuild_index))
			compare_start = static_cast<size_t>(start_rebuild_index);
	} else {
		compare_start = static_cast<size_t>(start_rebuild_index);
	}



	// Calcola media e deviazione standard
	double ref_sum = 0.0, ref_sq_sum = 0.0;
	double sig_sum = 0.0, sig_sq_sum = 0.0;
	size_t n = len - compare_start;

	for (size_t t = compare_start; t < len; ++t) {
		ref_sum += reference_signal[t];
		ref_sq_sum += reference_signal[t] * reference_signal[t];

		sig_sum += signal[t];
		sig_sq_sum += signal[t] * signal[t];
	}

	double ref_mean = ref_sum / n;
	double ref_std = std::sqrt(ref_sq_sum / n - ref_mean * ref_mean);
	double sig_mean = sig_sum / n;
	double sig_std = std::sqrt(sig_sq_sum / n - sig_mean * sig_mean);

	// Calcola errore sui segnali normalizzati e moltiplicati per 100
	for (size_t t = compare_start; t < len; ++t) {
		double ref_norm = ((reference_signal[t] - ref_mean) / ref_std) * 100.0;
		double sig_norm = ((signal[t] - sig_mean) / sig_std) * 100.0;
		double err = ref_norm - sig_norm;
		mse += err * err;
	}



    return -mse / len;  // negate MSE so higher is better
}



std::vector<double> run_genetic_algorithm(
    std::vector<double> reference_signal,
    int population_n,
    double CXPB,
    double MUTPB,
    int NGEN,
    int gene_length,
    std::optional<std::vector<double>> lb = std::nullopt,
    std::optional<std::vector<double>> ub = std::nullopt,
    int steps_n = 100,
    std::optional<std::vector<double>> initial_vector_opt = std::nullopt,
    std::optional<int> n_cycles_opt = std::nullopt,
    bool genetic_elitism = true,
    int elitism_elements = 10,
    std::optional<bool> initial_random_amplitudes_opt = std::nullopt,
    bool optimize_amplitudes = true,
    bool optimize_frequencies = true,
    bool optimize_phases = true,
    int start_rebuild_index = 0,
    bool period_related_rebuild_range = false,
    double period_multiplier = 1.0
)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_01(0.0, 1.0);
    std::uniform_int_distribution<> step_dist(0, steps_n - 1);

    auto generate_discrete_value = [&](int i) -> double {
        if (lb && ub) {
            double l = (*lb)[i];
            double u = (*ub)[i];
            double step_size = (u - l) / (steps_n - 1);
            int k = step_dist(gen);
            return l + k * step_size;
        } else {
            return dis_01(gen);
        }
    };

    int n_cycles = n_cycles_opt.value_or(1);
    int block_size = gene_length / n_cycles;

    std::vector<std::vector<double>> population;
	
	for (int i = 0; i < population_n; ++i) {
		std::vector<double> ind(gene_length);

		bool random_amplitudes = initial_random_amplitudes_opt.value_or(false);

		if (!initial_vector_opt || initial_vector_opt->size() != static_cast<size_t>(gene_length)) {
			throw std::runtime_error("initial_vector_opt must be provided with correct size");
		}

		// Ampiezze
		for (int c = 0; c < n_cycles; ++c) {
			if (random_amplitudes)
				ind[c] = generate_discrete_value(c);
			else
				ind[c] = (*initial_vector_opt)[c];
		}

		// Frequenze
		for (int c = 0; c < n_cycles; ++c)
			ind[n_cycles + c] = (*initial_vector_opt)[n_cycles + c];

		// Fasi
		for (int c = 0; c < n_cycles; ++c)
			ind[2 * n_cycles + c] = (*initial_vector_opt)[2 * n_cycles + c];

		population.push_back(ind);
	}


    std::vector<double> best_individual;
    double best_fitness = -1e9;

	std::vector<std::pair<double, std::vector<double>>> evaluated(population.size());
	std::vector<double> hall_of_fame;
	double hall_of_fame_fitness = -1e9;

	for (int gen_index = 0; gen_index < NGEN; ++gen_index) {
		#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(population.size()); ++i) {
//			evaluated[i] = std::make_pair(evaluate_fitness(population[i]), population[i]);
			evaluated[i] = std::make_pair(
				evaluate_fitness(population[i], reference_signal, n_cycles, start_rebuild_index, period_related_rebuild_range, period_multiplier),
				population[i]
			);

		}

		std::sort(evaluated.begin(), evaluated.end(), [](auto& a, auto& b) {
			return a.first > b.first;
		});

		if (evaluated[0].first > hall_of_fame_fitness) {
			hall_of_fame_fitness = evaluated[0].first;
			hall_of_fame = evaluated[0].second;
		}
		
		// DEBUG
		if (gen_index % 1000 == 0 || gen_index == NGEN - 1) {
			std::cout << "[DEBUG GEN " << gen_index << "] Best fitness: " << evaluated[0].first << std::endl;
		}


		std::vector<std::vector<double>> offspring;

		if (genetic_elitism) {
			for (int i = 0; i < elitism_elements && i < static_cast<int>(evaluated.size()); ++i) {
				offspring.push_back(evaluated[i].second);
			}
		}

		while (offspring.size() < population_n) {
			std::uniform_int_distribution<> pick(0, static_cast<int>(evaluated.size()) - 1);
			auto parent1 = evaluated[pick(gen)].second;
			auto parent2 = evaluated[pick(gen)].second;

			std::vector<double> child1 = parent1;
			std::vector<double> child2 = parent2;

			for (int i = 0; i < gene_length; i += block_size) {
				if (dis_01(gen) < CXPB) {
					for (int j = 0; j < block_size; ++j)
						if (i + j < gene_length)
							std::swap(child1[i + j], child2[i + j]);
				}
			}

			// MUTATION: only mutate blocks that are enabled by optimization flags
			auto mutate_block = [&](std::vector<double>& child, int offset, bool enabled) {
				if (!enabled) return;
				if (dis_01(gen) < MUTPB) {
					for (int j = 0; j < n_cycles; ++j)
						if (offset + j < gene_length)
							child[offset + j] = generate_discrete_value(offset + j);
				}
			};

			// mutate always amplitudes, conditionally others
			mutate_block(child1, 0, true);
			mutate_block(child1, n_cycles, optimize_frequencies);
			mutate_block(child1, 2 * n_cycles, optimize_phases);

			mutate_block(child2, 0, true);
			mutate_block(child2, n_cycles, optimize_frequencies);
			mutate_block(child2, 2 * n_cycles, optimize_phases);



			offspring.push_back(child1);
			if (offspring.size() < population_n)
				offspring.push_back(child2);
		}

		population = std::move(offspring);
	}


    return hall_of_fame;
}

PYBIND11_MODULE(cyGAoptMultiCore, m) {
	m.def("run_genetic_algorithm", &run_genetic_algorithm,
		py::arg("reference_signal"),
		py::arg("population_n"),
		py::arg("CXPB"),
		py::arg("MUTPB"),
		py::arg("NGEN"),
		py::arg("gene_length"),
		py::arg("lb") = std::nullopt,
		py::arg("ub") = std::nullopt,
		py::arg("steps_n") = 100,
		py::arg("initial_vector_opt") = std::nullopt,
		py::arg("n_cycles_opt") = std::nullopt,
		py::arg("genetic_elitism") = true,
		py::arg("elitism_elements") = 10,
		py::arg("initial_random_amplitudes") = false,
		py::arg("optimize_amplitudes") = true,
		py::arg("optimize_frequencies") = true,
		py::arg("optimize_phases") = true,
		py::arg("start_rebuild_index") = 0,
		py::arg("period_related_rebuild_range") = false,
		py::arg("period_multiplier") = 1.0

	);
}


