# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
screenShotHeight = 320
screenShotWidth = 320

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 80
maskHeight = 200

# Autoaim mouse movement amplifier
aaMovementAmp = .4

# Person Class Confidence
confidence = 0.4

# What key to press to quit and shutdown the autoaim
aaQuitKey = "Q"

# If you want to main slightly upwards towards the head
headshot_mode = True

# Displays the Corrections per second in the terminal
cpsDisplay = True

# Set to True if you want to get the visuals
visuals = False

# Smarter selection of people
centerOfScreen = True

# ONNX ONLY - Choose 1 of the 3 below
# 1 - CPU
# 2 - AMD
# 3 - NVIDIA
onnxChoice = 1
# ========================================  
# Enhanced Features by yin1895
# ========================================

# ========== Driver Mouse Configuration ==========
useDriverMouse = True          # Use driver-level control (requires Logitech GHUB/LGS)
fallbackToSystemMouse = True    # Fallback to system-level control when driver fails

# ========== Advanced Aiming Configuration ==========  
enableAdvancedAiming = False    # Enable advanced aiming system
enableSmoothMovement = True     # Enable smooth movement
smoothingFactor = 0.6           # Smoothing factor (0.1-0.9)
movementThreshold = 2           # Movement threshold (pixels)
enablePredictiveAim = False     # Enable predictive aiming
predictionStrength = 0.6        # Prediction strength
