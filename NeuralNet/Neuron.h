#pragma once
#include <vector>

class Neuron
{
private:
	double _input = 0.0f;
	double _value = 0.0f;
	double _error = 0.0f;
	std::vector<double> _weights;

public:

	Neuron(double value);

	void Activate();

	void AddWeight(double weight);

	std::vector<double> GetAllWeights();

	double GetWeightAt(int index);

	void SetWeights(std::vector<double> weights);

	double GetValue();

	void SetValue(double value);

	double GetError();

	void SetError(double error);

	void Sigmoid();

	void AdjustWeights(double learningRate);

	double SigmoidTransferFunctionDerivative(double output);
};