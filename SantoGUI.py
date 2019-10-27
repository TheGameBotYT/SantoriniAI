import numpy as np
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import NumericProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from SantoGame import SantoriniEnv

env_instance = SantoriniEnv(mode='play')  # Dummy


class SantoGUI(FloatLayout):
    # TODO: Show viable actions with a slight hue (non-player color)

    def __init__(self, env):
        super(SantoGUI, self).__init__()
        self.env = env
        self.x_dict = {0: 0.25, 1: 0.50, 2: 0.75, 3: 0.25, 4: 0.50, 5: 0.75, 6: 0.25, 7: 0.50, 8: 0.75}
        self.y_dict = {0: 2/3, 1: 2/3, 2: 2/3, 3: 1/3, 4: 1/3, 5: 1/3, 6: 0, 7: 0, 8: 0}
        for button_nr in range(9):
            x_pos = self.x_dict[button_nr]
            y_pos = self.y_dict[button_nr]
            self.add_widget(SantoButton(env, id=str(button_nr), pos_hint={'x': x_pos, 'y': y_pos}))
        self.run = Clock.schedule_interval(self.update, 2.0/60.0)

    def update(self, dt):
        """
        Paint job:
        For every square:
        - If a viable action can be done here -> green
        - If player 1 here -> Red
        - If player 2 here -> Blue
        :param dt:
        :return:
        """
        self.paint_position_viability()
        self.paint_building_level()

    def paint_building_level(self):
        for button in self.children:
            build_state_str = 'Build' + button.id
            image_str = 'Level' + str(self.env.state[build_state_str]) + '.png'
            button.source = image_str

    def paint_position_viability(self):
        viable_actions = self.env.get_viable_actions()
        viable_coords = [self.env.target_position_given_action(a) for a in viable_actions]
        viable_buttons = [str(self.env.inv_coords[c]) for c in viable_coords]
        p1_button, p2_button = self.env.player_positions[1], self.env.player_positions[2]
        for button in self.children:
            if button.id in viable_buttons:
                button.color = [0.5, 1, 0.5, 1]
            elif button.id == str(p1_button):
                button.color = [1, 0.5, 0.5, 1]
            elif button.id == str(p2_button):
                button.color = [0.5, 0.5, 0.5, 1]
            else:
                button.color = [1, 1, 1, 1]


class SantoButton(ButtonBehavior, Image):

    def __init__(self, env, **kwargs):
        self.env = env
        super(SantoButton, self).__init__(**kwargs)
        self.font_size = 30
        self.color = [1, 1, 1, 1]
        self.size_hint = (0.25, 1/3)
        self.source = 'Level0.png'

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            target_pos = int(self.id)
            action = self.env.action_given_target_position(target_pos)
            _, _, done = self.env.step(action)
            if done:
                pass  # TODO: Switch to victory/loss screen


class CustomScreenManager(ScreenManager):
    score = NumericProperty(0)


class TitleScreen(Screen):
    def __init__(self, name):
        super(TitleScreen, self).__init__()
        self.add_widget(Button(text='Click Here'))
        self.name = name

    def on_enter(self, *args):
        self.__init__(name='TITLE')

    def on_leave(self, *args):
        self.clear_widgets()

    def on_touch_down(self, touch):
        self.manager.current = 'GAME'


sm = CustomScreenManager()
sm.add_widget(TitleScreen(name='TITLE'))
game = SantoGUI(env_instance)
game_screen = Screen(name='GAME')
game_screen.add_widget(game)
sm.add_widget(game_screen)


class SantoGUIApp(App):

    def build(self):
        return sm


app = SantoGUIApp()
app.run()