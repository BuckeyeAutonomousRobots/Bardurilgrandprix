import asyncio
from mavsdk import System

async def run():
    # Initialize the drone
    drone = System()
    
    # Connect to the SITL instance
    # "udp://:14540" is the standard port for offboard APIs in PX4
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Global position estimate OK")
            break

    async for terrain_info in drone.telemetry.home():
        home_altitude = terrain_info.absolute_altitude_m
        break

    target_alt_amsl = home_altitude + 100.0

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()

    # Wait for 5 seconds to observe the takeoff
    await asyncio.sleep(5)

    await drone.action.goto_location(42.397606, 8.543060, target_alt_amsl, 0.0)

    # await asyncio.sleep(15)

    # print("-- Landing")
    # await drone.action.land()

if __name__ == "__main__":
    # Run the async loop
    asyncio.run(run())