extends Node3D

@export var slerp_speed: float = 20.0

var udp := PacketPeerUDP.new()
var filter := MadgwickAHRS.new()
var target_quaternion := Quaternion.IDENTITY

@onready var mesh = $MeshInstance3D


func _ready():
	udp.close()
	udp.bind(5005) 
	print("Listening for 32-byte packets on 5005. Madgwick Filter Active!")

func _process(delta: float):
	while udp.get_available_packet_count() > 0:
		var pkt = udp.get_packet()
		
		# 8 variables * 4 bytes = 32 bytes
		if pkt.size() == 32:
			
			# Decode all 6 IMU floats from their exact byte offsets
			var ax = pkt.decode_float(8)
			var ay = pkt.decode_float(12)
			var az = pkt.decode_float(16)
			var gx = pkt.decode_float(20)
			var gy = pkt.decode_float(24)
			var gz = pkt.decode_float(28)
			
			# Run 6-DOF Madgwick algorithm (dt = 0.01s for 100Hz)
			var raw_q = filter.update_6dof(gx, gy, gz, ax, ay, az, 0.01)
			
			# Map IMU orientation into Godot coordinate space (Y-up)
			# Catch NaNs to prevent the grey blob of death
			if not is_nan(raw_q.x):
				target_quaternion = Quaternion(raw_q.x, raw_q.z, -raw_q.y, raw_q.w)

	# Smoothly rotate the mesh to match the target quaternion
	if mesh:
		mesh.quaternion = mesh.quaternion.slerp(target_quaternion, slerp_speed * delta)
