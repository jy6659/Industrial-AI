# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:44:37 2021

@author: Jhung Joon Young
"""

from durable.lang import *
from durable.lang import _main_host

if _main_host is not None:
    _main_host._ruleset_directory.clear()
   
with ruleset('device'):

    @when_all(( m.name != "") & (m.power != "") & (m.wiring != "") & 
              (m.communication != "") & (m.controller != ""))  #(+m.name)
    def fact(c):
        print('Added devie -> name: "{0}", power: {1}, wiring: {2},'
              ' communication: {3}, controller: {4}'.format(c.m.name,
                                                            c.m.power, 
                                                            c.m.wiring, 
                                                            c.m.communication, 
                                                            c.m.controller))
        
    #motor
    @when_all(c.origin << (m.name == 'motor') & (m.status == 'error'),
              (m.power != 24) & (m.name == c.origin.name))
    def motor_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'motor') & (m.status == 'error'),
              (m.wiring != 10) & (m.name == c.origin.name))
    def motor_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'motor') & (m.status == 'error'),
              (m.communication != 'RS485') & (m.name == c.origin.name))
    def motor_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})       
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'motor') & (m.status == 'error'),
              (m.controller != 'PC') & (m.name == c.origin.name))
    def motor_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))        
           
    #lidar
    @when_all(c.origin << (m.name == 'lidar') & (m.status == 'error'),
              (m.power != 12) & (m.name == c.origin.name))
    def lidar_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'lidar') & (m.status == 'error'),
              (m.wiring != 2) & (m.name == c.origin.name))
    def lidar_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'lidar') & (m.status == 'error'),
              (m.communication != 'ethernet') & (m.name == c.origin.name))
    def lidar_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})        
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'lidar') & (m.status == 'error'),
              (m.controller != 'PC') & (m.name == c.origin.name))
    def lidar_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))        
     
    #ultrasona
    @when_all(c.origin << (m.name == 'ultrasona') & (m.status == 'error'),
              (m.power != 12) & (m.name == c.origin.name))
    def ultrasona_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'ultrasona') & (m.status == 'error'),
              (m.wiring != 4) & (m.name == c.origin.name))
    def ultrasona_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'ultrasona') & (m.status == 'error'),
              (m.communication != 'RS232') & (m.name == c.origin.name))
    def ultrasona_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})       
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'ultrasona') & (m.status == 'error'),
              (m.controller != 'MCU') & (m.name == c.origin.name))
    def ultrasona_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))        
        
    #bumper
    @when_all(c.origin << (m.name == 'bumper') & (m.status == 'error'),
              (m.power != 12) & (m.name == c.origin.name))
    def bumper_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'bumper') & (m.status == 'error'),
              (m.wiring != 3) & (m.name == c.origin.name))
    def bumper_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'bumper') & (m.status == 'error'),
              (m.communication != 'signal') & (m.name == c.origin.name))
    def bumper_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})       
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'bumper') & (m.status == 'error'),
              (m.controller != 'MCU') & (m.name == c.origin.name))
    def bumper_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))        
        
    #dynamixel
    @when_all(c.origin << (m.name == 'dynamixel') & (m.status == 'error'),
              (m.power != 12) & (m.name == c.origin.name))
    def dynamixel_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'dynamixel') & (m.status == 'error'),
              (m.wiring != 4) & (m.name == c.origin.name))
    def dynamixel_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'dynamixel') & (m.status == 'error'),
              (m.communication != 'RS485') & (m.name == c.origin.name))
    def dynamixel_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})        
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'dynamixel') & (m.status == 'error'),
              (m.controller != 'MCU') & (m.name == c.origin.name))
    def dynamixel_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))
        
    #neopixel
    @when_all(c.origin << (m.name == 'neopixel') & (m.status == 'error'),
              (m.power != 5) & (m.name == c.origin.name))
    def neopixel_power_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'power'})        
        print('Fact; {0} power error '.format(c.origin.name))
        
    @when_all(c.origin << (m.name == 'neopixel') & (m.status == 'error'),
              (m.wiring != 3) & (m.name == c.origin.name))
    def neopixel_wiring_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'wiring'})        
        print('Fact; {0} wiring error '.format(c.origin.name))
    
    @when_all(c.origin << (m.name == 'neopixel') & (m.status == 'error'),
              (m.communication != 'signal') & (m.name == c.origin.name))
    def neopixel_communication_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'change', 'object': 'interface'})        
        print('Fact; {0} communication error '.format(c.origin.name))        
    
    @when_all(c.origin << (m.name == 'neopixel') & (m.status == 'error'),
              (m.controller != 'MCU') & (m.name == c.origin.name))
    def neopixel_controller_error(c):        
        c.assert_fact({'name': c.origin.name, 'fix': 'reboot', 'object': 'S/W'})        
        print('Fact; {0} controller error '.format(c.origin.name))
     
    @when_all(+m.fix)
    def output(c):
        print('Repair; {0} {1} {2}'.format(c.m.fix, c.m.name, c.m.object))
        
# =============================================================================
# Device 상태 추가 
# =============================================================================
print('\n-------------------- Add Dvice Status --------------------\n')
try:
    assert_fact('device', {
        'name': 'motor',
        'power': 12,    #error
        'wiring': 5, 
        'communication': 'RS485',
        'controller': 'PC'
    })
    assert_fact('device', {
        'name': 'lidar',
        'power': 12, 
        'wiring': 5,    #error 
        'communication': 'ethernet',
        'controller': 'PC'
    })
    assert_fact('device', {
        'name': 'ultrasona',
        'power': 12, 
        'wiring': 4, 
        'communication': 'ethernet',    #error
        'controller': 'MCU'
    })
    assert_fact('device', {
        'name': 'bumper',
        'power': 12, 
        'wiring': 3, 
        'communication': 'signal',
        'controller': 'PC'  #error
    })
    assert_fact('device', {
        'name': 'dynamixel',
        'power': 24,    #error 
        'wiring': 4, 
        'communication': 'RS485',
        'controller': 'MCU'
    })
    assert_fact('device', {
        'name': 'neopixel',
        'power': 5, 
        'wiring': 1,    #error 
        'communication': 'signal',
        'controller': 'MCU'
    })
except BaseException as e:
    print('Error: {0}'.format(e.message))

# =============================================================================
# 고장 상태 추가     
# =============================================================================
print('\n-------------------- Add Error Status --------------------\n')
post('device', {'name': 'motor', 'status': 'error'})
post('device', {'name': 'lidar', 'status': 'error'})
post('device', {'name': 'ultrasona', 'status': 'error'})
post('device', {'name': 'bumper', 'status': 'error'})
post('device', {'name': 'dynamixel', 'status': 'error'})
post('device', {'name': 'neopixel', 'status': 'error'})
    
