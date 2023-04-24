# Wind turbine operating mode prediction

Predict the state of the wind turbine based on telemetry data.

Implements the "Convolutional neural network fault classification based on time series
analysis for benchmark wind turbine machine" paper by Rahimilarki, Gao, Jin, and Zhang
(published 2022 in "Renewable Energy").

## Background

In modern industrial applications, sensors are used to observe key machine characteristics.
These can help detect slight deviations and issues to avoid major system failures through
targeted repairs while keeping maintenance costs under control.

Such sensors are critical in the operation of wind turbines. Due to fluctuating winds that can
negatively impact the turbines and the high costs of maintenance, specifically for offshore
turbines, sensor readings need to be reliably converted into an operating mode. That is, given
the sensor readings over time, we want to know whether the turbine is operating correctly or
whether one of several issues is present. If issues are detected (reliably), targeted efforts can be
made to alleviate them before a major system failure occurs.

The aim of this project is to predict the operating mode of a wind turbine based on the
time series data from two sensors.

## Dataset

The data represent sensor readings and operating modes for 4,000 turbine runs.

The dataset comes in 3 pickle files:

- `time_series_1.pkl` and `time_series_2.pkl` are NumPy arrays of shape (4000,5000).
Each observation corresponds to 5,000 sensor readings from a turbine over time by one of
the two sensors (`time_series_1` measures the pitch angle in each second of operation,
and `time_series_2` measures the generator torque).

- `y.pkl` is the operating mode for each of the 4,000 turbine runs, in which
  - 0 if the turbine is healthy
  - 1 if the generator torque is faulty
  - 2 if the pitch angle is faulty, and
  - 3 if both are faulty.

The dataset is balanced in that each operating mode is represented equally often. The
overall objective is to predict the operating mode of a turbine as a function of the
sensor readings

## Report

The report can be found [here](./Report.md)
