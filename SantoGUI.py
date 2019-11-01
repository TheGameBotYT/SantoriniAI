import numpy as np
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import NumericProperty, BooleanProperty, StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from SantoGame import SantoriniEnv
from kivy.core.window import Window
Window.size = (600, 800)
Window.clearcolor = (1, 1, 1, 1)
import time

class SantoGUI(FloatLayout):
    # TODO: Don't crash when choosing a non-viable action
    # TODO: Make a label above for the current player

    def __init__(self, env):
        super(SantoGUI, self).__init__()
        self.env = env
        self.x_dict = {0: 0, 1: 1/3, 2: 2/3, 3: 0, 4: 1/3, 5: 2/3, 6: 0, 7: 1/3, 8: 2/3}
        self.y_dict = {0: 0, 1: 0, 2: 0, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.5, 7: 0.5, 8: 0.5}
        for button_nr in range(9):
            x_pos = self.x_dict[button_nr]
            y_pos = self.y_dict[button_nr]
            self.add_widget(SantoButton(env, id=str(button_nr), pos_hint={'x': x_pos, 'y': y_pos}))
        self.add_widget(Label(id='CurrentPlayerLabel', pos_hint={'x': 0, 'y': 0.75},
                               font_size=30))
        self.run = Clock.schedule_interval(self.update, 2.0 / 60.0)

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
        if self.env.end_condition()[0]:
            self.run.cancel()
            if self.env.end_condition()[1] == 1:
                self.parent.manager.winner = 'Victory \n Click to play again!'
            elif self.env.end_condition()[1] == -1:
                self.parent.manager.winner = 'Loss \n Click to play again!'
            self.parent.manager.current = 'END'

    def paint_building_level(self):
        for child_widget in self.children:
            if child_widget.id != 'CurrentPlayerLabel':
                build_state_str = 'Build' + child_widget.id
                image_str = 'Level' + str(self.env.state[build_state_str]) + '.png'
                child_widget.source = image_str
            else:
                print(child_widget.text)

    def paint_position_viability(self):
        viable_actions = self.env.get_viable_actions()
        viable_coords = [self.env.target_position_given_action(a) for a in viable_actions]
        viable_buttons = [str(self.env.inv_coords[c]) for c in viable_coords]
        p1_button, p2_button = self.env.player_positions[1], self.env.player_positions[2]
        for child_widget in self.children:
            if child_widget.id != 'CurrentPlayerLabel':
                if child_widget.id in viable_buttons:
                    child_widget.color = [0.5, 1, 0.5, 1]
                elif child_widget.id == str(p1_button):
                    child_widget.color = [1, 0.5, 0.5, 1]
                elif child_widget.id == str(p2_button):
                    child_widget.color = [0.5, 0.5, 0.5, 1]
                else:
                    child_widget.color = [1, 1, 1, 1]


class SantoButton(ButtonBehavior, Image):

    def __init__(self, env, **kwargs):
        self.env = env
        super(SantoButton, self).__init__(**kwargs)
        self.font_size = 30
        self.color = [1, 1, 1, 1]
        self.size_hint = (1/3, 0.25)
        self.source = 'Level0.png'

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            target_pos = int(self.id)
            action = self.env.action_given_target_position(target_pos)
            if action is None:
                pass
            else:
                _, _, done = self.env.step(action)


class CustomScreenManager(ScreenManager):
    winner = StringProperty(True)


class TitleScreen(Screen):
    def __init__(self, name):
        super(TitleScreen, self).__init__()
        self.name = name

    def on_enter(self, *args):
        self.add_widget(Button(text='Click To Play'))

    def on_leave(self, *args):
        self.clear_widgets()

    def on_touch_down(self, touch):
        self.manager.current = 'GAME'


class GameScreen(Screen):
    text = StringProperty('')

    def __init__(self, name):
        super(GameScreen, self).__init__()
        self.name = name

    def on_enter(self, *args):
        self.add_widget(gui)

    def on_leave(self, *args):
        self.clear_widgets()


class EndScreen(Screen):
    def __init__(self, name):
        super(EndScreen, self).__init__()
        self.name = name

    def on_enter(self, *args):
        text = self.parent.winner
        self.add_widget(Button(text=text))

    def on_leave(self, *args):
        self.clear_widgets()

    def on_touch_down(self, touch):
        self.manager.current = 'GAME'

env_instance = SantoriniEnv(mode='play')
gui = SantoGUI(env_instance)

sm = CustomScreenManager()
sm.add_widget(TitleScreen(name='TITLE'))
sm.add_widget(GameScreen(name='GAME'))
sm.add_widget(EndScreen(name='END'))
sm.current = 'TITLE'

class SantoGUIApp(App):

    def build(self):
        return sm


app = SantoGUIApp()
app.run()