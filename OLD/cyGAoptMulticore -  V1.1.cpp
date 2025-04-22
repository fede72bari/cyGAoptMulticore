#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <algorithm>
#include <optional>
#include <cmath>
#include <iostream>
#include <numeric>

namespace py = pybind11;

std::vector<double> run_genetic_algorithm(
    py::object fitness_func,
    int population_n,
    double CXPB,
    double MUTPB,
    int NGEN,
    int gene_length,
    std::optional<std::vector<double>> lb = std::nullopt,
    std::optional<std::vector<double>> ub = std::nullopt,
    int steps_n = 100,
    std::optional<std::vector<double>> initial_vector_opt = std::nullopt,
    std::optional<int> n_cycles_opt = std::nullopt
) {
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

	auto evaluate_population = [&](std::vector<std::vector<double>>& pop) {
		std::vector<std::pair<double, std::vector<double>>> evaluated(pop.size());

		#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
			double score = fitness_func(pop[i]).cast<double>();
			evaluated[i] = std::make_pair(score, pop[i]);
		}

		return evaluated;
	};


    auto tournament_selection = [&](const std::vector<std::pair<double, std::vector<double>>>& scored, int k = 3) {
        std::uniform_int_distribution<> pick(0, static_cast<int>(scored.size()) - 1);
        auto best = scored[pick(gen)];
        for (int i = 1; i < k; ++i) {
            auto candidate = scored[pick(gen)];
            if (candidate.first > best.first) best = candidate;
        }
        return best.second;
    };

    int n_cycles = n_cycles_opt.value_or(1);
    int block_size = gene_length / n_cycles;
//    std::cout << "[DEBUG] gene_length: " << gene_length << ", n_cycles: " << n_cycles << ", block_size: " << block_size << std::endl;

    std::vector<std::vector<double>> population;
    for (int i = 0; i < population_n; ++i) {
        std::vector<double> ind(gene_length);
        int include_freqs = (gene_length >= 2 * n_cycles);
        int include_phases = (gene_length >= 3 * n_cycles);

//        std::cout << "[DEBUG] Generating individual " << i << std::endl;

        for (int c = 0; c < n_cycles; ++c) {
            ind[c] = generate_discrete_value(c);  // Ampiezze random
        }

        if (initial_vector_opt) {
            size_t vec_len = initial_vector_opt->size();
//            std::cout << "[DEBUG] initial_vector_opt size: " << vec_len << std::endl;

            if (vec_len != static_cast<size_t>(gene_length)) {
                throw std::runtime_error("initial_vector_opt size does not match gene_length");
            }

            if (vec_len >= static_cast<size_t>(2 * n_cycles)) {
                for (int c = 0; c < n_cycles; ++c) {
                    ind[n_cycles + c] = (*initial_vector_opt)[n_cycles + c];
                }
            }

            if (vec_len == static_cast<size_t>(3 * n_cycles)) {
                for (int c = 0; c < n_cycles; ++c) {
                    ind[2 * n_cycles + c] = (*initial_vector_opt)[2 * n_cycles + c];
                }
            }
        }

//        std::cout << "[DEBUG] Generated individual: ";
        for (double val : ind) std::cout << val << " ";
        std::cout << std::endl;

        population.push_back(ind);
    }

    std::vector<double> best_individual;
    double best_fitness = -1e9;

    for (int gen_index = 0; gen_index < NGEN; ++gen_index) {
        auto evaluated = evaluate_population(population);

        std::sort(evaluated.begin(), evaluated.end(), [](auto& a, auto& b) { return a.first > b.first; });

        best_individual = evaluated[0].second;
        best_fitness = evaluated[0].first;

        std::vector<std::vector<double>> offspring;
        while (offspring.size() < population_n) {
            auto parent1 = tournament_selection(evaluated);
            auto parent2 = tournament_selection(evaluated);

            std::vector<double> child1 = parent1;
            std::vector<double> child2 = parent2;

            for (int i = 0; i < gene_length; i += block_size) {
                if (dis_01(gen) < CXPB) {
                    for (int j = 0; j < block_size; ++j) {
                        if (i + j < gene_length)
                            std::swap(child1[i + j], child2[i + j]);
                    }
                }
            }

            for (int i = 0; i < gene_length; i += block_size) {
                if (dis_01(gen) < MUTPB) {
                    for (int j = 0; j < block_size; ++j) {
                        if (i + j < gene_length)
                            child1[i + j] = generate_discrete_value(i + j);
                    }
                }
                if (dis_01(gen) < MUTPB) {
                    for (int j = 0; j < block_size; ++j) {
                        if (i + j < gene_length)
                            child2[i + j] = generate_discrete_value(i + j);
                    }
                }
            }

            offspring.push_back(child1);
            if (offspring.size() < population_n)
                offspring.push_back(child2);
        }

        population = std::move(offspring);
    }

    return best_individual;
}

PYBIND11_MODULE(cyGAoptMultiCore, m) {
    m.def("run_genetic_algorithm", &run_genetic_algorithm,
        py::arg("fitness_func"),
        py::arg("population_n"),
        py::arg("CXPB"),
        py::arg("MUTPB"),
        py::arg("NGEN"),
        py::arg("gene_length"),
        py::arg("lb") = std::nullopt,
        py::arg("ub") = std::nullopt,
        py::arg("steps_n") = 100,
        py::arg("initial_vector_opt") = std::nullopt,
        py::arg("n_cycles_opt") = std::nullopt
    );
}
