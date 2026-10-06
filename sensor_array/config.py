"""Physical constants and default settings for the array-estimation problem."""

C_SOUND = 1540.0                # speed of sound in seawater (m/s)
SENSOR_Z = 20.0                 # sensor depth (m)
TX_Z = 2.0                      # transmitter depth (m)
DZ = SENSOR_Z - TX_Z            # vertical separation (m) = 18
N_SENSORS = 1926                # number of sensors on the cable
N_TX = 6                        # number of transmitter locations
MAX_SPACING = 1.0213            # max cable length between consecutive sensors (m)

# Optimisation defaults
LAMBDA_TRACK = 1e-7             # weight of the weak ship-track prior
HUBER_DELTA = 0.005             # Huber transition (s) ~ 7.7 m of range error
MAXITER = 2000
MAXFUN = 200_000
