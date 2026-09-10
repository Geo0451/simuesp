class_name MadgwickAHRS

var beta: float = 0.1 # Algorithm gain (0.04 - 0.1 for stylus control)
var q: Quaternion = Quaternion.IDENTITY

func update_6dof(gx_deg: float, gy_deg: float, gz_deg: float, ax: float, ay: float, az: float, dt: float) -> Quaternion:
	# Convert gyro from deg/s to rad/s
	var gx = deg_to_rad(gx_deg)
	var gy = deg_to_rad(gy_deg)
	var gz = deg_to_rad(gz_deg)
	
	var q1 = q.w; var q2 = q.x; var q3 = q.y; var q4 = q.z
	
	# Compute rate of change of quaternion from gyroscope
	var qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz)
	var qDot2 = 0.5 * ( q1 * gx + q3 * gz - q4 * gy)
	var qDot3 = 0.5 * ( q1 * gy - q2 * gz + q4 * gx)
	var qDot4 = 0.5 * ( q1 * gz + q2 * gy - q3 * gx)

	# Compute feedback only if accelerometer measurement is valid
	var accel_len = sqrt(ax * ax + ay * ay + az * az)
	if accel_len > 0.001:
		ax /= accel_len
		ay /= accel_len
		az /= accel_len
		
		# Auxiliary variables to avoid repeated calculations
		var _2q1 = 2.0 * q1
		var _2q2 = 2.0 * q2
		var _2q3 = 2.0 * q3
		var _2q4 = 2.0 * q4
		var _4q1 = 4.0 * q1
		var _4q2 = 4.0 * q2
		var _4q3 = 4.0 * q3
		var _8q2 = 8.0 * q2
		var _8q3 = 8.0 * q3
		var q1q1 = q1 * q1
		var q2q2 = q2 * q2
		var q3q3 = q3 * q3
		var q4q4 = q4 * q4

		# Gradient descent step magnitude and direction
		var s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
		var s2 = _4q2 * q4q4 - _2q4 * ax + 4.0 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
		var s3 = 4.0 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
		var s4 = 4.0 * q2q2 * q4 - _2q2 * ax + 4.0 * q3q3 * q4 - _2q3 * ay

		var s_len = sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
		if s_len > 0.001:
			s1 /= s_len; s2 /= s_len; s3 /= s_len; s4 /= s_len

		# Apply feedback step
		qDot1 -= beta * s1
		qDot2 -= beta * s2
		qDot3 -= beta * s3
		qDot4 -= beta * s4

	# Integrate rate of change of quaternion to yield quaternion
	q1 += qDot1 * dt
	q2 += qDot2 * dt
	q3 += qDot3 * dt
	q4 += qDot4 * dt

	q = Quaternion(q2, q3, q4, q1).normalized()
	return q
