#pragma once

enum class NeuronType
{
	INPUT,
	HIDDEN,
	OUTPUT
};

class Neuron
{
private:
	double _bias = 0.0f;
	double _weight = 0.0f;
	double _value = 0.0f;

	NeuronType _type;

public:

	Neuron(double value, NeuronType type, double bias = Randf(), double weight = Randf());

	void Activate();

	void SetBias(double bias);

	double GetBias();

	void SetWeight(double weight);

	double GetWeight();

	void SetValue(double value);

	double GetValue();

	NeuronType GetType();

	static double Randf();

	double Sigmoid(double value);

	double TransferFunctionDerivative(double output);
};