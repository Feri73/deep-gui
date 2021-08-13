EMMA                 �      \(    ' org/jtb/alogcat/Format org/jtb/alogcat Format�&)�#�g9 Format.java   	 values ()[Lorg/jtb/alogcat/Format;                       valueOf ,(Ljava/lang/String;)Lorg/jtb/alogcat/Format;                       <init> B(Ljava/lang/String;ILjava/lang/String;ILjava/util/regex/Pattern;)V                   6   5   4   3   7   3 getTitle -(Landroid/content/Context;)Ljava/lang/String;                   :   : byValue ,(Ljava/lang/String;)Lorg/jtb/alogcat/Format;                   >   > getLevel +(Ljava/lang/String;)Lorg/jtb/alogcat/Level;                               B      C      F   E      G      I   B 
getByOrder (I)Lorg/jtb/alogcat/Format;                   M   M getValue ()Ljava/lang/String;                   Q   Q <clinit> ()V          �         -   ,   +   *   )   (   '   &   %   "                                                          org/jtb/alogcat/LogActivity org/jtb/alogcat LogActivityY�4_u�. LogActivity.java   * <init> ()V                   K   I   '   E   ' jumpTop ()V                   a   \   [   [ 
jumpBottom ()V                   f   e   d   d cat (Ljava/lang/String;)V                   
               i      j      m   o   n      p      r      w   v   u   i cat (Ljava/util/List;)V                            z      z      |   {   z      ~   }   z onCreate (Landroid/os/Bundle;)V          +      	   �   �   �   �   �   �   �   �   �   � onNewIntent (Landroid/content/Intent;)V                                     �      �      �      �      �   �      �      �   � onStart ()V                   �   �   � init ()V          #      	   �   �   �   �   �   �   �   �   �   � onResume ()V          	         �   �   �   �   � onPause ()V                   �   �   � onStop ()V                         �   �      �      �   � 	onDestroy ()V                   �   �   � onSaveInstanceState (Landroid/os/Bundle;)V                   �   � onRestoreInstanceState (Landroid/os/Bundle;)V                   �   � reset ()V          
               �   �   �      �      �     �   � onCreateOptionsMenu (Landroid/view/Menu;)Z          x                    
    1  .  -  +  (  '  &  #  "  !           onPrepareOptionsMenu (Landroid/view/Menu;)Z                  6  6 setPlayMenu ()V             
                    :     ;     =     >  ?     B  A     D  : setFilterMenu ()V                                    G     H     L  K     L     M     O     R  Q  G onOptionsItemSelected (Landroid/view/MenuItem;)Z                                                V     Y  X     \  [     _  ^  b  a     d     e     g     i     m  l  k     q  p  o     s  V onActivityResult (IILandroid/content/Intent;)V                        y     {     ~  y setKeepScreenOn ()V                           �     �     �     �  � onContextItemSelected (Landroid/view/MenuItem;)Z             	   	           �     �  �  �     �  �  �     �  � clear ()V                           �  �     �  �  �     �     �  � dump (Z)Ljava/lang/String;       
                     
            
     �  �  �  �     �     �  �     �  �     �  �     �     �     �  �  �  �  �     �     �  � share ()V                  �  �  � save ()Ljava/io/File;          &        �    �  �  � onCreateDialog (I)Landroid/app/Dialog;             	                
  	        pauseLog ()V                
                                           playLog ()V                   
                        !        #  "     %     (  '   
access$000 0(Lorg/jtb/alogcat/LogActivity;Ljava/util/List;)V                   '   ' 
access$100 @(Lorg/jtb/alogcat/LogActivity;)Lorg/jtb/alogcat/LogEntryAdapter;                   '   ' 
access$200 8(Lorg/jtb/alogcat/LogActivity;)Landroid/widget/ListView;                   '   ' 
access$300  (Lorg/jtb/alogcat/LogActivity;)V                   '   ' 
access$402 O(Lorg/jtb/alogcat/LogActivity;Lorg/jtb/alogcat/Logcat;)Lorg/jtb/alogcat/Logcat;                   '   ' 
access$500 <(Lorg/jtb/alogcat/LogActivity;)Lorg/jtb/alogcat/LogActivity;                   '   ' 
access$600 3(Lorg/jtb/alogcat/LogActivity;)Landroid/os/Handler;                   '   ' 
access$400 7(Lorg/jtb/alogcat/LogActivity;)Lorg/jtb/alogcat/Logcat;                   '   ' 
access$700 6(Lorg/jtb/alogcat/LogActivity;)Lorg/jtb/alogcat/Prefs;                   '   ' 
access$800 2(Lorg/jtb/alogcat/LogActivity;Z)Ljava/lang/String;                   '   ' <clinit> ()V                   *   (   ( Eorg/jtb/alogcat/EmmaInstrument/InstrumentedActivity$CoverageCollector org/jtb/alogcat/EmmaInstrument &InstrumentedActivity$CoverageCollector�ҡ���G InstrumentedActivity.java    <init> 8(Lorg/jtb/alogcat/EmmaInstrument/InstrumentedActivity;)V                       	onReceive 4(Landroid/content/Context;Landroid/content/Intent;)V                      	                                        (   '   #   !      (      )      .    org/jtb/alogcat/LogActivity$7 org/jtb/alogcat LogActivity$7��Ϗ��f LogActivity.java    <init> <(Lorg/jtb/alogcat/LogActivity;Ljava/io/File;Ljava/io/File;)V                  �  � run ()V                                              	        �  �     �     �  �  �  �  �     �  �     �  �  �     �  �  �     �  �     �  �  �     �     �  �     �  �     �        � org/jtb/alogcat/Lock org/jtb/alogcat LockN��3���
 	Lock.java    <init> ()V                       getLock =(Landroid/content/Context;)Landroid/os/PowerManager$WakeLock;                                               acquire (Landroid/content/Context;)V                                            release ()V                                               $      %      (      ,    org/jtb/alogcat/Logcat$1 org/jtb/alogcat Logcat$1��f�gM� Logcat.java    <init> (Lorg/jtb/alogcat/Logcat;)V                   $   $ run ()V          	      
               (      )      +   ,      -      1   0   /   ( org/jtb/alogcat/LogEntryAdapter org/jtb/alogcat LogEntryAdapter|���,��� LogEntryAdapter.java    <init> *(Landroid/app/Activity;ILjava/util/List;)V                                   getView A(ILandroid/view/View;Landroid/view/ViewGroup;)Landroid/view/View;                                      #   "   !      $      )   (   '   ,    remove (I)V          
         2   1   0   0 areAllItemsEnabled ()Z                   5   5 	isEnabled (I)Z                   9   9 get (I)Lorg/jtb/alogcat/LogEntry;                   =   = 
getEntries ()Ljava/util/List;                   A   A org/jtb/alogcat/LogActivity$2 org/jtb/alogcat LogActivity$2;�D�n��� LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                   \   \ run ()V                   _   ^   ^ org/jtb/alogcat/LogActivity$1 org/jtb/alogcat LogActivity$1��^xcy.� LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                   K   K handleMessage (Landroid/os/Message;)V                	            N      R   Q   P      T      W   N org/jtb/alogcat/LogActivity$4 org/jtb/alogcat LogActivity$4qJ�K��` LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                   �   � onScrollStateChanged  (Landroid/widget/AbsListView;I)V                   �   �   � onScroll "(Landroid/widget/AbsListView;III)V                   �   � org/jtb/alogcat/LogActivity$3 org/jtb/alogcat LogActivity$3;�D���c� LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                   �   � onCreateContextMenu Z(Landroid/view/ContextMenu;Landroid/view/View;Landroid/view/ContextMenu$ContextMenuInfo;)V                   �   �   �   �   �   � org/jtb/alogcat/PrefsActivity org/jtb/alogcat PrefsActivityt���ȡ PrefsActivity.java   
 <init> ()V                       onCreate (Landroid/os/Bundle;)V          3      
   (   '   $   "                       setLevelTitle ()V                   +   ,   + setFormatTitle ()V                   0   /   / setBufferTitle ()V                   4   3   3 setTextsizeTitle ()V                   8   7   7 setBackgroundColorTitle ()V                   <   ;   ; onResume ()V                   @   J   H   F   E   D   C   B   @ onPause ()V                   Q   O   N   N onSharedPreferenceChanged 8(Landroid/content/SharedPreferences;Ljava/lang/String;)V                                                 U      V      W      X      Y      Z      [      \      ]      ^      `   U !org/jtb/alogcat/BackgroundColor$1 org/jtb/alogcat BackgroundColor$1  k�7�� BackgroundColor.java    <init> ()V                                   org/jtb/alogcat/LogActivity$6 org/jtb/alogcat LogActivity$6dPu)�:� LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                  �  � run ()V          	                          �  �  �  �     �     �     �  �     �     �     �  �  �  � org/jtb/alogcat/LogActivity$5 org/jtb/alogcat LogActivity$5;�D�=�Z LogActivity.java    <init>  (Lorg/jtb/alogcat/LogActivity;)V                   �   � run ()V                        �   � org/jtb/alogcat/BackgroundColor org/jtb/alogcat BackgroundColor-�[��� BackgroundColor.java    values $()[Lorg/jtb/alogcat/BackgroundColor;                       valueOf 5(Ljava/lang/String;)Lorg/jtb/alogcat/BackgroundColor;                       <init> )(Ljava/lang/String;IILjava/lang/String;)V                             !    valueOfHexColor 5(Ljava/lang/String;)Lorg/jtb/alogcat/BackgroundColor;                   $   $ getColor ()I                   (   ( getTitle -(Landroid/content/Context;)Ljava/lang/String;                   ,   , <clinit> ()V          ,                         org/jtb/alogcat/LogDumper org/jtb/alogcat 	LogDumper�6ƚ�&�? LogDumper.java    <init> (Landroid/content/Context;)V          	                   dump (Z)Ljava/lang/String;                                           
                                 	                  L                  &      $   #         (      )      )      +      .      0   /      3   2      4      6      =   ;   :   8      @   ?   E      J   G      I   H      L      M      M      A   E   C   B      J   G      I   H      L      M      M      E      J   G      I   H      L      M      M    org/jtb/alogcat/ShareService org/jtb/alogcat ShareServiceL�;���� ShareService.java    <init> ()V                          onHandleIntent (Landroid/content/Intent;)V                                                                      &   "      &      &      +   *   (   &   .   -    org/jtb/alogcat/FilterDialog$2 org/jtb/alogcat FilterDialog$2P�\?��� FilterDialog.java    <init> l(Lorg/jtb/alogcat/FilterDialog;Landroid/widget/EditText;Landroid/widget/CheckBox;Landroid/widget/TextView;)V                   A   A onClick %(Landroid/content/DialogInterface;I)V          %   
               E   D   C      L   G      K   J   I   H      W   V   U   S   R   P   O   X   C org/jtb/alogcat/Logcat org/jtb/alogcat Logcat�*�_�&. Logcat.java    <init> 0(Landroid/content/Context;Landroid/os/Handler;)V          G         $   #   "   !                  C   B   A   @   ?   >   =   ;   9   8   7   7 start ()V       %                                       	                                                                  !   
   2   %      X   W   V   U   S   Q   P   L   K   I   G      Z   Y      a   ^   \      e      e      f      g      i      j      l      m      m      o      r      r      u      x   z   y      z      z      �      �   �      �      �   �   �      �   �   �      �   }   |      �   �      �      �   �   �      �   �      �      �      �   �      �      �   �   �      �   �      �      �   G cat ()V                      	            �      �   �      �   �   �   �      �      �      �   � stop ()V                            �   �      �      �   �      �   � 	isRunning ()Z                   �   � isPlay ()Z                   �   � setPlay (Z)V                   �   �   � 
access$000 (Lorg/jtb/alogcat/Logcat;)Z                       
access$100 (Lorg/jtb/alogcat/Logcat;)J                       
access$102 (Lorg/jtb/alogcat/Logcat;J)J                       
access$200 (Lorg/jtb/alogcat/Logcat;)V                       org/jtb/alogcat/FilterDialog$3 org/jtb/alogcat FilterDialog$3F��mq�� FilterDialog.java    <init> S(Lorg/jtb/alogcat/FilterDialog;Landroid/widget/EditText;Landroid/widget/CheckBox;)V                   [   [ onClick %(Landroid/content/DialogInterface;I)V          '      
   j   i   h   g   e   c   b   `   _   ]   ] org/jtb/alogcat/SaveService org/jtb/alogcat SaveService8������ SaveService.java    <init> ()V                   
   	   	 onHandleIntent (Landroid/content/Intent;)V          
                      org/jtb/alogcat/FilterDialog$4 org/jtb/alogcat FilterDialog$4F��k�vB FilterDialog.java    <init> S(Lorg/jtb/alogcat/FilterDialog;Landroid/widget/EditText;Landroid/widget/CheckBox;)V                   m   m onClick %(Landroid/content/DialogInterface;I)V                   v   u   t   r   q   o   o "org/jtb/alogcat/ALogcatApplication org/jtb/alogcat ALogcatApplication6�|b�Q�O ALogcatApplication.java    <init> ()V                       onCreate ()V                          org/jtb/alogcat/Prefs org/jtb/alogcat Prefs�$��B�-� 
Prefs.java    <init> (Landroid/content/Context;)V          
                      	getString 8(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;                          	setString '(Ljava/lang/String;Ljava/lang/String;)V                   &   %   $   #   # 
getBoolean (Ljava/lang/String;Z)Z                   *   )   ) 
setBoolean (Ljava/lang/String;Z)V                   1   0   /   .   . getLevel ()Lorg/jtb/alogcat/Level;                   4   4 setLevel (Lorg/jtb/alogcat/Level;)V                   9   8   8 	getFormat ()Lorg/jtb/alogcat/Format;                
         A   <      C   B      F   < 	setFormat (Lorg/jtb/alogcat/Format;)V                   K   J   J 	getBuffer ()Lorg/jtb/alogcat/Buffer;                   N   N 	setBuffer (Lorg/jtb/alogcat/Buffer;)V                   S   R   R getTextsize ()Lorg/jtb/alogcat/Textsize;                   V   V setTextsize (Lorg/jtb/alogcat/Textsize;)V                   [   Z   Z 	getFilter ()Ljava/lang/String;                   ^   ^ getFilterPattern ()Ljava/util/regex/Pattern;                                  b      c      g   f      h      k      m   l   o   n   b 	setFilter (Ljava/lang/String;)V                   u   t   t getBackgroundColor #()Lorg/jtb/alogcat/BackgroundColor;                      	         x      |      ~   }      �      �      �   x isShareHtml ()Z                   �   �   � isKeepScreenOn ()Z                   �   �   � setKeepScreenOn (Z)V                   �   �   � isFilterPattern ()Z                   �   � setFilterPattern (Z)V                   �   �   � org/jtb/alogcat/LogEntry org/jtb/alogcat LogEntry�v��1��& LogEntry.java    <init> ,(Ljava/lang/String;Lorg/jtb/alogcat/Level;)V                   
   	                   getLevel ()Lorg/jtb/alogcat/Level;                       getText ()Ljava/lang/String;                       hashCode ()I       	                        
      	                                                                      equals (Ljava/lang/Object;)Z                                                          $      %      &      '      (      )      +   *      ,      -      .      /      0      1      2   $ org/jtb/alogcat/FilterDialog$1 org/jtb/alogcat FilterDialog$16NkSA�� FilterDialog.java    <init> :(Lorg/jtb/alogcat/FilterDialog;Landroid/widget/TextView;)V          	         0   0 onCheckedChanged #(Landroid/widget/CompoundButton;Z)V             	            4      6   5      8   4 org/jtb/alogcat/FilterDialog org/jtb/alogcat FilterDialog���ɭ�+� FilterDialog.java    dismiss ()V                                         <init>  (Lorg/jtb/alogcat/LogActivity;)V          p         -   +   Z   *   (   &   $   #   !          y      @   >   =   l   ;   0   /    
access$002 "(Lorg/jtb/alogcat/FilterDialog;Z)Z                       
access$100 7(Lorg/jtb/alogcat/FilterDialog;)Lorg/jtb/alogcat/Prefs;                       
access$200 =(Lorg/jtb/alogcat/FilterDialog;)Lorg/jtb/alogcat/LogActivity;                       org/jtb/alogcat/SaveReceiver org/jtb/alogcat SaveReceiver6�|d0��� SaveReceiver.java    <init> ()V                       	onReceive 4(Landroid/content/Context;Landroid/content/Intent;)V                                   6org/jtb/alogcat/EmmaInstrument/SMSInstrumentedReceiver org/jtb/alogcat/EmmaInstrument SMSInstrumentedReceiverx��\{5w SMSInstrumentedReceiver.java    <init> ()V                       	onReceive 4(Landroid/content/Context;Landroid/content/Intent;)V                	                           &    <clinit> ()V                       3org/jtb/alogcat/EmmaInstrument/InstrumentedActivity org/jtb/alogcat/EmmaInstrument InstrumentedActivity�#��@�6 InstrumentedActivity.java    <init> ()V                   
      
 setFinishListener 2(Lorg/jtb/alogcat/EmmaInstrument/FinishListener;)V                       finish ()V                         6   5   4      7      9   4 
access$000 f(Lorg/jtb/alogcat/EmmaInstrument/InstrumentedActivity;)Lorg/jtb/alogcat/EmmaInstrument/FinishListener;                   
   
 <clinit> ()V                       org/jtb/alogcat/Level org/jtb/alogcat Level�/bxR 
Level.java   	 values ()[Lorg/jtb/alogcat/Level;                       valueOf +(Ljava/lang/String;)Lorg/jtb/alogcat/Level;                       <init> *(Ljava/lang/String;IILjava/lang/String;I)V                                "   !    getHexColor ()Ljava/lang/String;                   %   % getColor ()I                   )   ) getValue ()I                   -   - 
getByOrder (I)Lorg/jtb/alogcat/Level;                   1   1 getTitle -(Landroid/content/Context;)Ljava/lang/String;                   5   5 <clinit> ()V          m                                    
   	          org/jtb/alogcat/Intent org/jtb/alogcat IntentW���c�� Intent.java    <init> ()V                       handleExtras 4(Landroid/content/Context;Landroid/content/Intent;)V                                                                       org/jtb/alogcat/LogSaver org/jtb/alogcat LogSaver@���#�� LogSaver.java    <init> (Landroid/content/Context;)V                                   save ()Ljava/io/File;          .         )   &   #   D   !   ! 
access$000 7(Lorg/jtb/alogcat/LogSaver;)Lorg/jtb/alogcat/LogDumper;                       <clinit> ()V                          2org/jtb/alogcat/EmmaInstrument/EmmaInstrumentation org/jtb/alogcat/EmmaInstrument EmmaInstrumentation�Qwj
MG EmmaInstrumentation.java    <init> ()V                            !    onCreate (Landroid/os/Bundle;)V                         (   &   %      *   )      0   /   .   -   % onStart ()V                   6   5   ;   :   9   8   5 getBooleanArgument ((Landroid/os/Bundle;Ljava/lang/String;)Z                               ?   >      ?      ?      ?      ?   > generateCoverageReport ()V                               5         a   T   G   Q   E   P      V   a   U      W   a   X      a   Z   Y      a   \   [      a   ^   ]      `   _      b   E getCoverageFilePath ()Ljava/lang/String;                         e      f      h   e setCoverageFilePath (Ljava/lang/String;)Z                            m      m      o   n      q   m reportEmmaError (Ljava/lang/Exception;)V                   v   u   u reportEmmaError *(Ljava/lang/String;Ljava/lang/Exception;)V                   }   {   z   y   y onActivityFinished ()V                         �   �      �      �   �   � dumpIntermediateCoverage (Ljava/lang/String;)V                               �   �      �      �      �   �      �   � <clinit> ()V                       org/jtb/alogcat/Textsize org/jtb/alogcat Textsize�m��^m~ Textsize.java    values ()[Lorg/jtb/alogcat/Textsize;                       valueOf .(Ljava/lang/String;)Lorg/jtb/alogcat/Textsize;                       <init> *(Ljava/lang/String;ILjava/lang/Integer;I)V                       #   "   !     getTitle -(Landroid/content/Context;)Ljava/lang/String;                   &   & 
getByOrder (I)Lorg/jtb/alogcat/Textsize;                   *   * getValue ()Ljava/lang/Integer;                   .   . <clinit> ()V          P                           
   	                  	 org/jtb/alogcat/Buffer org/jtb/alogcat Bufferbכ�;<�� Buffer.java    values ()[Lorg/jtb/alogcat/Buffer;                   	   	 valueOf ,(Ljava/lang/String;)Lorg/jtb/alogcat/Buffer;                   	   	 <init> )(Ljava/lang/String;ILjava/lang/String;I)V                   $   #   "   !   ! getTitle -(Landroid/content/Context;)Ljava/lang/String;                   '   ' byValue ,(Ljava/lang/String;)Lorg/jtb/alogcat/Buffer;                   +   + 
getByOrder (I)Lorg/jtb/alogcat/Buffer;                   /   / getValue ()Ljava/lang/String;                   3   3 <clinit> ()V          M                              
   	               
 org/jtb/alogcat/LogSaver$1 org/jtb/alogcat 
LogSaver$1���=�F} LogSaver.java    <init> 9(Lorg/jtb/alogcat/LogSaver;Ljava/io/File;Ljava/io/File;)V                   )   ) run ()V                                              
         +   -      .      5   4   3   1   9      >   ;      >   =   <      6   9   7      >   ;      >   =   <      9      >   ;      =   <      >      A   + org/jtb/alogcat/ShareReceiver org/jtb/alogcat ShareReceiver6�|d0��� ShareReceiver.java    <init> ()V                       	onReceive 4(Landroid/content/Context;Landroid/content/Intent;)V                                  