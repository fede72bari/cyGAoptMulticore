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

namespace py = pybind11;

// FITNESS PURAMENTE C++
double evaluate_fitness(const std::vector<double>& individual) {
    double sum = 0.0;
    for (double val : individual) {
        sum += val * val;
    }
    return -sum;  // esempio: funzione da minimizzare
}

std::vector<double> run_genetic_algorithm(
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
    int elitism_elements = 10
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

    int n_cycles = n_cycles_opt.value_or(1);
    int block_size = gene_length / n_cycles;

    std::vector<std::vector<double>> population;
    for (int i = 0; i < population_n; ++i) {
        std::vector<double> ind(gene_length);

        for (int c = 0; c < n_cycles; ++c) {
            ind[c] = generate_discrete_value(c);
        }

        if (initial_vector_opt) {
            size_t vec_len = initial_vector_opt->size();

            if (vec_len != static_cast<size_t>(gene_length)) {
                throw std::runtime_error("initial_vector_opt size does not match gene_length");
            }

            if (vec_len >= static_cast<size_t>(2 * n_cycles)) {
                for (int c = 0; c < n_cycles; ++c)
                    ind[n_cycles + c] = (*initial_vector_opt)[n_cycles + c];
            }

            if (vec_len == static_cast<size_t>(3 * n_cycles)) {
                for (int c = 0; c < n_cycles; ++c)
                    ind[2 * n_cycles + c] = (*initial_vector_opt)[2 * n_cycles + c];
            }
        }

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
			evaluated[i] = std::make_pair(evaluate_fitness(population[i]), population[i]);
		}

		std::sort(evaluated.begin(), evaluated.end(), [](auto& a, auto& b) {
			return a.first > b.first;
		});

		if (evaluated[0].first > hall_of_fame_fitness) {
			hall_of_fame_fitness = evaluated[0].first;
			hall_of_fame = evaluated[0].second;
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

			for (int i = 0; i < gene_length; i += block_size) {
				if (dis_01(gen) < MUTPB) {
					for (int j = 0; j < block_size; ++j)
						if (i + j < gene_length)
							child1[i + j] = generate_discrete_value(i + j);
				}
				if (dis_01(gen) < MUTPB) {
					for (int j = 0; j < block_size; ++j)
						if (i + j < gene_length)
							child2[i + j] = generate_discrete_value(i + j);
				}
			}

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
		py::arg("elitism_elements") = 10
    );
}
