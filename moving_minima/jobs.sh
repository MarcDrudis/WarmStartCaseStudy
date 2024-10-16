#!/bin/bash


seq 4 16 | parallel 'echo {}'
# seq 4 16 | parallel 'python follow_adiabatic_minima.py {} 5'

